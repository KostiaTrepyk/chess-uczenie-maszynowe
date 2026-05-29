import torch
import torch.nn as nn

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
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        # Pierwsza warstwa splotowa (konwolucyjna) - szuka lokalnych wzorców na planszy (np. obrona bierek)
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        # Normalizacja warstwy (Batch Normalization) - stabilizuje i znacznie przyspiesza proces uczenia
        self.bn1 = nn.BatchNorm2d(channels)
        
        # Druga warstwa splotowa - buduje bardziej złożone cechy na podstawie wyników pierwszej warstwy
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        
        # Funkcja aktywacji LeakyReLU - zapobiega problemowi "umierających neuronów" (przepuszcza mały gradient dla wartości < 0)
        self.leaky_relu = nn.LeakyReLU(0.1)

        # Blok Squeeze-and-Excitation (SE) - mechanizm uwagi (attention) dla kanałów.
        # Pozwala sieci "zrozumieć", które mapy cech są w danym momencie najważniejsze.
        self.se = nn.Sequential(
            # Squeeze (Ściskanie): globalna kompresja przestrzenna z 8x8 na 1x1. Tworzy ogólne podsumowanie dla każdego kanału.
            nn.AdaptiveAvgPool2d(1), 
            # Excitation (Wzbudzanie) krok 1: redukcja wymiarowości kanałów w celu zmniejszenia kosztów obliczeniowych
            nn.Conv2d(channels, channels // reduction, kernel_size=1),
            nn.LeakyReLU(0.1),
            # Excitation krok 2: przywrócenie oryginalnej liczby kanałów
            nn.Conv2d(channels // reduction, channels, kernel_size=1),
            # Funkcja Sigmoid skaluje wartości do przedziału [0, 1] - działa jak bramka wyliczająca wagę (znaczenie) każdego kanału
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Zapisujemy oryginalne wejście (tzw. skip connection / połączenie rezydualne)
        residual = x

        # Główne przejście sygnału przez konwolucje
        out = self.leaky_relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        # Zastosowanie mechanizmu uwagi
        # Przemnażamy uzyskane cechy przez wyliczone wagi (od 0 do 1) z bloku SE
        out = out * self.se(out)

        # Kluczowy element ResNet: dodajemy oryginalne, nieprzetworzone wejście do wyniku.
        # Zapobiega to degradacji dokładności i znikaniu gradientu w bardzo głębokich sieciach.
        out += residual
        
        # Ostateczna aktywacja warstwy
        return self.leaky_relu(out)

class ChessResNet(nn.Module):
    def __init__(self, num_blocks=10):
        super().__init__()
        
        # Blok wejściowy. 
        # Zmienia początkową reprezentację planszy (15 warstw wejściowych: bierki, kolej ruchu, roszada, en passant) 
        # na bogatszą reprezentację wielowymiarową (256 kanałów cech).
        self.input_conv = nn.Sequential(
            nn.Conv2d(15, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1)
        )

        # Główny rdzeń modelu (Backbone) oparty na architekturze ResNet.
        # Składa się z serii bloków rezydualnych (domyślnie 10). 
        # Służy do głębokiej ekstrakcji cech lokalnych (np. kontrola centrum, bezpieczeństwo króla).
        self.resnet_blocks = nn.Sequential(
            *[ResidualBlock(256) for _ in range(num_blocks)]
        )

        # <-- TUTAJ DODAJEMY TRANSFORMER -->
        # Blok Transformera dodaje globalny kontekst (globalny "ogląd" sytuacji).
        # Pozwala sieci natychmiast powiązać figury znajdujące się na przeciwległych końcach planszy.
        self.transformer = TransformerBlock(channels=256, heads=4)

        # Głowica oceniająca (Value Head). 
        # Przekształca wyodrębnione cechy w ostateczną ocenę pozycji.
        self.value_head = nn.Sequential(
            # Konwolucja 1x1 kompresuje liczbę kanałów z 256 na 16. 
            # Drastycznie zmniejsza to liczbę parametrów przed warstwą w pełni połączoną (Linear), oszczędzając pamięć.
            nn.Conv2d(256, 16, kernel_size=1), 
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.1),
            
            # Spłaszczanie danych z formatu 2D (kanały, wysokość, szerokość) do wektora 1D
            nn.Flatten(),
            
            # Główna warstwa decyzyjna z 512 neuronami
            nn.Linear(16 * 8 * 8, 512), 
            nn.LeakyReLU(0.1),
            
            # Zapobiega przeuczeniu (overfitting) poprzez losowe ignorowanie 30% neuronów podczas trenowania
            nn.Dropout(0.3),
            
            # Wyjście sieci. 
            # Sieć nie przewiduje jednej liczby, ale dokonuje klasyfikacji do jednego ze 100 "koszyków".
            # Koszyki te reprezentują ocenę pozycji w zakresie od -10.0 do +10.0 pionów.
            nn.Linear(512, 100),
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

def load_model()-> ChessResNet:
    # Wczytujemy gotowy model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 2. Tworzymy pustą architekturę sieci i przenosimy na urządzenie
    model = ChessResNet(num_blocks=10).to(device)

    # 3. Wczytujemy wagi z pliku
    # map_location gwarantuje poprawne wczytanie, nawet jeśli model był trenowany na GPU a uruchamiany na CPU
    model.load_state_dict(torch.load("best_chess_model.pth", map_location=device))

    # 4. Koniecznie przełączamy na tryb predykcji (eval)
    model.eval()

    return model