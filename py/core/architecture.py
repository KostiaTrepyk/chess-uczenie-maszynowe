import torch
import torch.nn as nn

from core.consts import num_blocks, channels, kernel_size, padding

class TransformerBlock(nn.Module):
    def __init__(self, channels: int, heads: int = 4):
        super().__init__()
        
        # Mechanizm wielogłowej uwagi (Multi-head Attention). 
        # batch_first=True oznacza oczekiwany format tensora: (Batch, Sequence, Features)
        self.attention = nn.MultiheadAttention(embed_dim=channels, num_heads=heads, batch_first=True)
        
        # Normalizacja warstwy w celu stabilizacji gradientów podczas trenowania
        self.norm = nn.LayerNorm(channels)
        
        # Trenowalne embeddingi pozycyjne dla 64 pól szachownicy.
        # Sam mechanizm uwagi nie zna układu pól (nie ma pamięci przestrzennej), 
        # dlatego dodajemy współrzędne (1 batch, 64 pola, wymiar kanałów).
        self.pos_embedding = nn.Parameter(torch.randn(1, 64, channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Tensor wejściowy z CNN ma wymiary: (Batch, Channels, Height=8, Width=8)
        b, c, h, w = x.size()
        
        # Przygotowanie danych dla transformera:
        # 1. Spłaszczamy planszę 2D (8x8) do sekwencji 1D o długości 64 (h * w)
        # 2. permute: zmieniamy kolejność wymiarów, aby uzyskać format -> (Batch, 64, Channels)
        x_flat = x.view(b, c, h * w).permute(0, 2, 1)
        
        # Dodajemy informację o pozycji do cech każdego pola
        x_flat = x_flat + self.pos_embedding
        
        # Zastosowanie mechanizmu Self-Attention (Q, K, V są tym samym tensorem)
        attn_out, _ = self.attention(x_flat, x_flat, x_flat)
        
        # Połączenie rezydualne (dodanie wejścia do wyjścia) + Layer Normalization
        out = self.norm(x_flat + attn_out)
        
        # Przekształcenie odwrotne:
        # 1. permute: przywracamy kanały na drugie miejsce -> (Batch, Channels, 64)
        # 2. view: przywracamy 64 pola z powrotem do formatu 2D -> (Batch, Channels, 8, 8)
        # Pozwala to na łatwą integrację bloku z pozostałymi warstwami splotowymi.
        return out.permute(0, 2, 1).view(b, c, h, w)

class ResidualBlock(nn.Module):
    def __init__(self, channels: int = 256):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        
        # Возвращаем твой LeakyReLU
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.1, inplace=True) 
        
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        
        # Наш новый умный эквалайзер каналов
        self.se = SEBlock(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        
        # Первая свертка + BatchNorm + Активация
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.leaky_relu(out)
        
        # Вторая свертка + BatchNorm
        out = self.conv2(out)
        out = self.bn2(out)
        
        # Применяем SEBlock ДО сложения с residual
        out = self.se(out)
        
        # Складываем и применяем финальную активацию
        out += residual
        out = self.leaky_relu(out)
        
        return out

class ChessResNet(nn.Module):
    def __init__(self, num_blocks=num_blocks, channels=channels):
        super().__init__()
        
        # Blok wejściowy. 
        # Zmienia początkową reprezentację planszy (15 warstw wejściowych: bierki, kolej ruchu, roszada, en passant) 
        # na bogatszą reprezentację wielowymiarową (256 kanałów cech).
        self.input_conv = nn.Sequential(
            nn.Conv2d(15, channels, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(channels),
            nn.LeakyReLU(0.1)
        )

        # Główny rdzeń modelu (Backbone) oparty na architekturze ResNet.
        # Składa się z serii bloków rezydualnych (domyślnie 10). 
        # Służy do głębokiej ekstrakcji cech lokalnych (np. kontrola centrum, bezpieczeństwo króla).
        self.resnet_blocks = nn.Sequential(
            *[ResidualBlock(channels) for _ in range(num_blocks)]
        )

        # <-- TUTAJ DODAJEMY TRANSFORMER -->
        # Blok Transformera dodaje globalny kontekst (globalny "ogląd" sytuacji).
        # Pozwala sieci natychmiast powiązać figury znajdujące się na przeciwległych końcach planszy.
        self.transformer = TransformerBlock(channels=channels, heads=8)

        # Głowica oceniająca (Value Head). 
        # Przekształca wyodrębnione cechy w ostateczną ocenę pozycji.
        self.value_head = nn.Sequential(
            # Konwolucja 1x1 kompresuje liczbę kanałów z 256 na 32. 
            # Drastycznie zmniejsza to liczbę parametrów przed warstwą w pełni połączoną (Linear), oszczędzając pamięć.
            nn.Conv2d(channels, 32, kernel_size=1), 
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1),
            
            # Spłaszczanie danych z formatu 2D (kanały, wysokość, szerokość) do wektora 1D
            nn.Flatten(),
            
            # Główna warstwa decyzyjna z 512 neuronami
            nn.Linear(32 * 8 * 8, 1024), 
            nn.LeakyReLU(0.1),
            
            # Zapobiega przeuczeniu (overfitting) poprzez losowe ignorowanie 30% neuronów podczas trenowania
            nn.Dropout(0.3),
            
            # Wyjście sieci. 
            # Sieć nie przewiduje jednej liczby, ale dokonuje klasyfikacji do jednego ze 100 "koszyków".
            # Koszyki te reprezentują ocenę pozycji w zakresie od -10.0 do +10.0 pionów.
            nn.Linear(1024, 100),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Przepływ sygnału (danych) przez kolejne etapy sieci:
        
        # 1. Wstępna obróbka z 15 na 256 kanałów
        x = self.input_conv(x)
        
        # 2. Głęboka analiza lokalnych struktur na planszy (konwolucje z uwagą SE)
        x = self.resnet_blocks(x)
        
        # 3. Dodanie globalnego kontekstu przez Transformer
        x = self.transformer(x) # <-- PRZEPUSZCZAMY PRZEZ TRANSFORMER
        
        # 4. Agregacja wyników i ostateczna predykcja (100 logitów dla każdego z koszyków)
        return self.value_head(x)

def load_model(num_blocks=num_blocks, channels=channels)-> ChessResNet:
    # Wczytujemy gotowy model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 2. Tworzymy pustą architekturę sieci i przenosimy na urządzenie
    model = ChessResNet(num_blocks, channels).to(device)

    # 3. Wczytujemy wagi z pliku
    # map_location gwarantuje poprawne wczytanie, nawet jeśli model był trenowany na GPU a uruchamiany na CPU
    print(f'Wczytywanie modelu z pliku: best_chess_model.pth')
    model.load_state_dict(torch.load("best_chess_model.pth", map_location=device))

    # 4. Koniecznie przełączamy na tryb predykcji (eval)
    model.eval()

    return model

class SEBlock(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super(SEBlock, self).__init__()
        # Сжимаем пространственную инфу до 1x1
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        
        # Используем свертки 1x1 вместо Linear (это гораздо надежнее в PyTorch)
        self.excitation = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x имеет форму [Batch, Channels, 8, 8]
        y = self.squeeze(x)        # -> [Batch, Channels, 1, 1]
        y = self.excitation(y)     # -> [Batch, Channels, 1, 1]
        
        # PyTorch сам автоматически "растянет" 1x1 до 8x8 при умножении (Broadcasting)
        return x * y
