import torch
import torch.nn as nn

from core.consts import NUM_BLOCKS, CHANNELS, KERNEL_SIZE, PADDING, TransformerBlockHeads

class TransformerBlock(nn.Module):
    # Blok odpowiedzialny za "globalne widzenie" - rozumie relacje między odległymi figurami
    def __init__(self, channels: int, heads: int = 16):
        super().__init__()
        # Mechanizm uwagi (Attention) - uczy się, które pola na szachownicy są zależne od innych
        self.attention = nn.MultiheadAttention(embed_dim=channels, num_heads=heads, batch_first=True)
        self.norm1 = nn.LayerNorm(channels) # Stabilizuje uczenie
        self.norm2 = nn.LayerNorm(channels)
        # Sieć gęsta (MLP) do przetwarzania cech po mechanizmie uwagi
        self.mlp = nn.Sequential(
            nn.Linear(channels, channels * 2),
            nn.GELU(), # Płynniejsza funkcja aktywacji niż ReLU
            nn.Linear(channels * 2, channels)
        )
        # Informacja o pozycji (aby Transformer wiedział, gdzie dokładnie na planszy 8x8 jest dana figura)
        self.pos_embedding = nn.Parameter(torch.randn(1, 64, channels) * 0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.size()
        # Spłaszczamy planszę 8x8 do 64 sekwencyjnych wektorów (jak słowa w zdaniu)
        x_flat = x.view(b, c, h * w).permute(0, 2, 1)
        x_flat = x_flat + self.pos_embedding # Dodajemy koordynaty pól
        
        norm_x1 = self.norm1(x_flat)
        attn_out, _ = self.attention(norm_x1, norm_x1, norm_x1) # Krok uwagi
        x_flat = x_flat + attn_out # Połączenie resztkowe (Residual connection)
        
        norm_x2 = self.norm2(x_flat)
        mlp_out = self.mlp(norm_x2)
        out = x_flat + mlp_out
        
        # Przywracamy format planszy 8x8
        return out.permute(0, 2, 1).view(b, c, h, w)

class ResidualBlock(nn.Module):
    # Klasyczny blok ResNet - wyłapuje lokalne motywy taktyczne (np. skoczek broniący piona)
    def __init__(self, channels: int = 256):
        super(ResidualBlock, self).__init__()
        # Dwie warstwy konwolucyjne czytające planszę
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.act = nn.SiLU(inplace=True) # SiLU (Swish) działa lepiej w głębokich sieciach
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        # Mechanizm skupienia uwagi na najważniejszych "kanałach" (np. zignoruj puste pola, patrz na króla)
        self.se = SEBlock(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x # Zapisujemy wejście
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)
        out.add_(residual) # Dodajemy wejście do wyjścia (omijanie - to zapobiega zanikaniu gradientu)
        out = self.act(out)
        return out

class ChessResNet(nn.Module):
    # Główny model hybrydowy (ResNet + Transformer)
    def __init__(self, num_blocks=NUM_BLOCKS, channels=CHANNELS, num_transformer_blocks: int = 3):
        super().__init__()
        # Początkowe przetworzenie 15-kanałowego wejścia (nasze figury, figury wroga, roszady itp.)
        self.input_conv = nn.Sequential(
            nn.Conv2d(15, channels, kernel_size=KERNEL_SIZE, padding=PADDING),
            nn.BatchNorm2d(channels),
            nn.LeakyReLU(0.1)
        )
        # Seria bloków ResNet (lokalna taktyka)
        self.resnet_blocks = nn.Sequential(*[ResidualBlock(channels) for _ in range(num_blocks)])
        # Seria bloków Transformera (globalna strategia)
        self.transformer = nn.Sequential(*[TransformerBlock(channels=channels, heads=TransformerBlockHeads) for _ in range(num_transformer_blocks)])
        
        # Zwężenie i spłaszczenie danych przed wysłaniem ich do "Głów"
        self.value_features = nn.Sequential(
            nn.Conv2d(channels, 4, kernel_size=1), # Redukcja z 256 do 4 kanałów
            nn.BatchNorm2d(4),
            nn.LeakyReLU(0.1),
            nn.Flatten(), # Zmiana z siatki 2D na wektor 1D
            nn.Linear(256, 512), 
            nn.LeakyReLU(0.1),
        )
        
        # --- ARCHITEKTURA DWÓCH GŁÓW (Two-Headed Output) ---
        # Głowa 1: Przewiduje punktową przewagę (+1.5, -2.0)
        self.score_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.LeakyReLU(0.1),
            nn.Linear(256, 1) # Zwraca jedną liczbę
        )
        
        # Głowa 2: Przewiduje szansę na mat (zwraca logity, które potem zamieniasz na % przez sigmoid)
        self.mate_head = nn.Sequential(
            nn.Linear(512, 128),
            nn.LeakyReLU(0.1),
            nn.Linear(128, 1) # Zwraca jedną liczbę
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.input_conv(x)
        x = self.resnet_blocks(x)
        x = self.transformer(x)
        
        features = self.value_features(x)
        
        # Model wyrzuca dwie niezależne oceny jednocześnie
        return self.score_head(features).squeeze(-1), self.mate_head(features).squeeze(-1)

def load_model(num_blocks=NUM_BLOCKS, channels=CHANNELS)-> ChessResNet:
    # Wczytywanie gotowego modelu z wagami
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = ChessResNet(num_blocks, channels).to(device)

    print(f'Wczytywanie modelu z pliku: best_chess_model.pth')
    # map_location bezpiecznie ładuje model niezależnie od tego, czy używasz CPU czy GPU
    model.load_state_dict(torch.load("best_chess_model.pth", map_location=device))

    # Wyłącza warstwy takie jak Dropout/BatchNorm, niezbędne podczas gry (Inference)
    model.eval()

    return model

class SEBlock(nn.Module):
    # Squeeze-and-Excitation - filtr, który wygasza szum i wzmacnia najważniejsze cechy planszy
    def __init__(self, channels: int, reduction: int = 16):
        super(SEBlock, self).__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1) # Squeeze: uśrednia każdy kanał do 1 piksela
        self.excitation = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, kernel_size=1, bias=False), # Zwężenie
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, kernel_size=1, bias=False), # Rozszerzenie
            nn.Sigmoid() # Excitation: nadaje wagi (od 0 do 1) dla każdego kanału
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.squeeze(x)
        y = self.excitation(y)
        return x * y # Mnoży oryginalną mapę przez wagi (wzmacnia dobre sygnały, tłumi złe)
