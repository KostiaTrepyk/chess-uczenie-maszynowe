import torch
import torch.nn as nn
from torch.amp import GradScaler
from torch.utils.data import DataLoader
from typing import List
from tqdm import tqdm
import torch.nn.functional as F

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

PIECE_TO_CHANNEL = {
    'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5,
    'p': 6, 'n': 7, 'b': 8, 'r': 9, 'q': 10, 'k': 11
}

# *** Konwersja FEN na tensor i trenowanie ***

def build_batch_on_gpu(boards, turns, castling, ep):
    """
    Błyskawicznie buduje wejściowy tensor (batch) o wymiarach [Batch, 15, 8, 8] 
    bezpośrednio na karcie graficznej (GPU). Jest to kluczowa optymalizacja wydajności,
    unikająca wąskiego gardła przy przesyłaniu gotowych, dużych tensorów z procesora (CPU).
    """
    # Pobieramy rozmiar aktualnego batcha (np. 1024 pozycje naraz)
    b = boards.size(0)
    device = boards.device
    
    # 1. Bierki (12 warstw) poprzez one-hot encoding
    # Zamieniamy indeksy figur na 13 klas (12 figur + puste pole), a następnie ucinamy 13-tą klasę (puste pola to zera).
    pieces = F.one_hot(boards, num_classes=13)[:, :, :12].float()
    # Zmieniamy kształt z (Batch, 64 pola, 12 figur) na (Batch, 12 figur, 8 wierszy, 8 kolumn)
    pieces = pieces.permute(0, 2, 1).view(b, 12, 8, 8)
    
    # 2. Kolej ruchu (1 warstwa)
    # Wypełniamy całą planszę 8x8 jedynkami (ruch białych) lub zerami (ruch czarnych).
    # expand() jest bardzo szybkie, bo nie kopiuje danych w pamięci, tylko tworzy wirtualny widok.
    turns_layer = turns.view(b, 1, 1, 1).expand(b, 1, 8, 8)
    
    # 3. Roszada (1 warstwa)
    castling_layer = torch.zeros((b, 1, 8, 8), device=device, dtype=torch.float32)
    # Kodujemy prawa do roszady ustawiając wartości '1' w rogach szachownicy:
    castling_layer[:, 0, 7, 7] = castling[:, 0] # Krótka roszada białych (h1)
    castling_layer[:, 0, 7, 0] = castling[:, 1] # Długa roszada białych (a1)
    castling_layer[:, 0, 0, 7] = castling[:, 2] # Krótka roszada czarnych (h8)
    castling_layer[:, 0, 0, 0] = castling[:, 3] # Długa roszada czarnych (a8)
    
    # 4. Bicie w przelocie - En Passant (1 warstwa)
    # Inicjalizujemy płaski tensor dla 64 pól.
    ep_layer = torch.zeros((b, 1, 64), device=device, dtype=torch.float32)
    
    # Wyszukujemy plansze w batchu, które w ogóle mają możliwość bicia w przelocie
    has_ep = ep > 0 
    ep_idx = ep[has_ep] - 1 # Przeliczamy indeksy z zakresu 1-64 na układ 0-63
    
    # Błyskawicznie wstawiamy jedynki tylko na odpowiednich polach (omijamy powolne pętle 'for')
    ep_layer[has_ep, 0, ep_idx] = 1.0 
    
    # Przekształcamy płaski tensor powrotem na układ planszy 8x8
    ep_layer = ep_layer.view(b, 1, 8, 8)
    
    # Ostatecznie sklejamy wszystkie warstwy w jedną macierz (wzdłuż wymiaru kanałów: dim=1).
    # 12 (bierki) + 1 (kolej ruchu) + 1 (roszada) + 1 (en passant) = 15 warstw.
    return torch.cat([pieces, turns_layer, castling_layer, ep_layer], dim=1)

def fen_to_tensor(fen: str) -> torch.Tensor:
    """
    Konwertuje tekstowy zapis pozycji (FEN) na tensor o wymiarach [15, 8, 8].
    Używane głównie do predykcji pojedynczych pozycji poza główną pętlą uczącą.
    """
    # Rozbijamy łańcuch FEN na poszczególne sekcje (oddzielone spacją)
    parts = fen.split()
    board_part = parts[0]
    turn_part = parts[1] if len(parts) > 1 else 'w'
    castling_part = parts[2] if len(parts) > 2 else '-'
    ep_part = parts[3] if len(parts) > 3 else '-' # <-- Odczytujemy En Passant

    # Inicjalizujemy tensor wypełniony zerami (TERAZ 15 WARSTW)
    # 12 na figury + 1 na ruch + 1 na roszadę + 1 na bicie w przelocie
    tensor = torch.zeros((15, 8, 8), dtype=torch.float32)

    # 1. Warstwy 0-11: Bierki
    row, col = 0, 0
    for char in board_part:
        if char == '/':
            # Znak '/' oznacza przejście do następnego wiersza na planszy
            row += 1
            col = 0
        elif char.isdigit():
            # Cyfra oznacza liczbę pustych pól z rzędu - przesuwamy wskaźnik kolumny
            col += int(char)
        else:
            # Rozpoznajemy figurę, pobieramy jej indeks (0-11) i wstawiamy 1.0 w odpowiednim miejscu
            channel = PIECE_TO_CHANNEL[char]
            tensor[channel, row, col] = 1.0
            col += 1

    # 2. Warstwa 12: Kolej ruchu
    # Jeśli ruch mają białe ('w'), wypełniamy całą warstwę 12 jedynkami. Dla czarnych zostają zera.
    if turn_part == 'w':
        tensor[12, :, :] = 1.0

    # 3. Warstwa 13: Roszada
    # Oznaczamy prawa do roszady ustawiając '1.0' w odpowiednich rogach warstwy 13
    if 'K' in castling_part: tensor[13, 7, 7] = 1.0 # Białe, krótka (pole h1)
    if 'Q' in castling_part: tensor[13, 7, 0] = 1.0 # Białe, długa (pole a1)
    if 'k' in castling_part: tensor[13, 0, 7] = 1.0 # Czarne, krótka (pole h8)
    if 'q' in castling_part: tensor[13, 0, 0] = 1.0 # Czarne, długa (pole a8)

    # 4. Warstwa 14: Bicie w przelocie (En Passant)
    if ep_part != '-':
        # Zamieniamy literę kolumny (np. 'a', 'e') na indeks numeryczny tablicy (0-7)
        ep_col = ord(ep_part[0]) - ord('a')
        
        # Zamieniamy cyfrę wiersza szachowego (1-8) na indeks tablicy (0-7). 
        # Szachowe wiersze liczy się od dołu, a tablice od góry, stąd odejmowanie od 8.
        ep_row = 8 - int(ep_part[1])
        
        # Zaznaczamy pole, na którym możliwe jest bicie w przelocie
        tensor[14, ep_row, ep_col] = 1.0

    return tensor

def train_model(train_loader : DataLoader, val_loader : DataLoader) -> ChessResNet:
    device_str = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nTrenowanie na urządzeniu: {device}")

    # Włączamy akcelerację sprzętową dla sieci splotowych (Daje +10-15% do szybkości)
    torch.backends.cudnn.benchmark = True

    model = ChessResNet(num_blocks=10).to(device)
    
    # Kompilacja modelu dla PyTorch 2.0+ (Daje jeszcze +20% do szybkości). W razie błędu na Windowsie - po prostu usuń tę linię.
    # model = torch.compile(model) 

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()
    epochs = 100

    patience = 5
    best_val_loss = float('inf')
    epochs_no_improve = 0

    scaler = GradScaler()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2
    )

    print("\nRozpoczynamy trenowanie...")
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        train_pbar = tqdm(train_loader, desc=f"Epoka {epoch+1}/{epochs} [Train]", leave=False)
        for batch_data in train_pbar:
            # 1. Przenosimy surowe i lekkie dane na kartę graficzną (GPU)
            boards, turns, castling, eps, batch_targets = [x.to(device, non_blocking=True) for x in batch_data]
            
            # 2. Błyskawicznie budujemy ciężkie macierze na GPU
            batch_inputs = build_batch_on_gpu(boards, turns, castling, eps)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(device_str):
                # PyTorch automatycznie zastosuje FlashAttention wewnątrz Transformera
                predictions = model(batch_inputs)
                loss = loss_fn(predictions, batch_targets)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()
            train_pbar.set_postfix({'loss': f"{loss.item():.4f}"})

        avg_train_loss = train_loss / len(train_loader)

        model.eval()
        val_loss = 0.0

        val_pbar = tqdm(val_loader, desc=f"Epoka {epoch+1}/{epochs} [Val]", leave=False)
        with torch.no_grad():
            for val_data in val_pbar:
                # To samo dla walidacji: przenosimy i budujemy
                boards, turns, castling, eps, val_targets = [x.to(device, non_blocking=True) for x in val_data]
                val_inputs = build_batch_on_gpu(boards, turns, castling, eps)

                val_preds = model(val_inputs)
                v_loss = loss_fn(val_preds, val_targets)
                val_loss += v_loss.item()
                
                val_pbar.set_postfix({'loss': f"{v_loss.item():.4f}"})

        avg_val_loss = val_loss / len(val_loader)

        # Zostawiamy tylko ostateczny wynik, paski znikną (leave=False)
        print(f"\tEpoka {epoch+1} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        scheduler.step(avg_val_loss)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), "best_chess_model.pth")
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print(f"\n🛑 WCZESNE ZATRZYMANIE (EARLY STOPPING)!")
            break

    model.load_state_dict(torch.load("best_chess_model.pth"))
    model.eval()
    print("Pomyślnie wczytano najlepszą wersję modelu.")
    
    return model

# *** Wczytywanie modelu i eksport ***

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

def export_model_to_onnx():
    device = torch.device('cpu')
    model = ChessResNet(num_blocks=10).to(device)
    model.load_state_dict(torch.load("best_chess_model.pth", map_location=device))
    model.eval()

    # Tworzymy fałszywy tensor o odpowiednim kształcie (1 batch, 15 warstw, 8x8)
    dummy_input = torch.randn(1, 15, 8, 8, device=device)

    # Eksportujemy
    torch.onnx.export(
        model,
        dummy_input,
        "chess_model.onnx",
        export_params=True,
        opset_version=18,
        input_names=['input'],
        output_names=['output']
    )
    print("✅ Plik chess_model.onnx jest gotowy!")

# *** Predykcja oceny ***

def predict_evaluation(model: nn.Module, fens: List[str]) -> List[float]:
    if not fens:
        return []

    model.eval()
    device = next(model.parameters()).device
    inputs = torch.stack([fen_to_tensor(fen) for fen in fens]).to(device)

    # Tablica wartości dla każdego koszyka od -10 do +10
    bucket_values = torch.linspace(-10.0, 10.0, steps=100, device=device)

    with torch.no_grad():
        # Pobieramy surowe logity
        logits = model(inputs)
        # Zamieniamy logity na rzeczywiste procenty prawdopodobieństwa (suma = 1.0)
        probabilities = torch.softmax(logits, dim=-1)
        
        # Wartość oczekiwana: Prawdopodobieństwo * Wartość koszyka
        # Daje to idealną ułamkową precyzję (np. 1.27 piona)
        expected_evals = torch.sum(probabilities * bucket_values, dim=-1).tolist()

    return expected_evals

def evaluate_model_metrics(model: torch.nn.Module, dataloader: torch.utils.data.DataLoader, device: torch.device) -> tuple[float, float]:
    model.eval()
    total_mae_pawns = 0.0
    correct_signs = 0
    total_positions = 0
    bucket_values = torch.linspace(-10.0, 10.0, steps=100, device=device)

    with torch.no_grad():
        for val_data in dataloader:
            boards, turns, castling, eps, batch_targets = [x.to(device, non_blocking=True) for x in val_data]
            
            # Budujemy batch na GPU
            batch_inputs = build_batch_on_gpu(boards, turns, castling, eps)

            # Przekazujemy do modelu
            logits = model(batch_inputs) 

            probabilities = torch.softmax(logits, dim=-1)
            preds_pawns = torch.sum(probabilities * bucket_values, dim=-1) 
            targets_pawns = bucket_values[batch_targets] 

            mae = torch.abs(preds_pawns - targets_pawns).sum().item()
            total_mae_pawns += mae

            preds_signs = torch.sign(preds_pawns)
            targets_signs = torch.sign(targets_pawns)
            correct_signs += (preds_signs == targets_signs).sum().item()

            total_positions += batch_targets.size(0)

    avg_mae_pawns = total_mae_pawns / total_positions
    sign_accuracy = (correct_signs / total_positions) * 100.0

    print(f"\n📊 Wyniki testowania ({total_positions} pozycji):")
    print(f"\tOcena przewagi (Sign Accuracy): {sign_accuracy:.1f}%")
    print(f"\tŚredni błąd (MAE):          {avg_mae_pawns:.2f} piona")

    return avg_mae_pawns, sign_accuracy

def show_model_stats(model, val_loader, df):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    test_positions = [
        [df["FEN"].iloc[i], df["Evaluation"].iloc[i]] for i in range(10)
    ]
    
    # 1. Wyciągamy tylko ciągi FEN dla sieci neuronowej
    just_fens = [item[0] for item in test_positions]

    # 2. Wykonujemy predykcję
    evaluations = predict_evaluation(model, just_fens)

    # 3. Wyświetlamy wynik
    print("\nWyniki oceny przez sieć neuronową:")
    for [fen, correct_eval], eval_score in zip(test_positions, evaluations):
        print(f"\nPozycja: {fen.split()[0]}")
        print(f"Ocena sieci: {eval_score:.2f} piona (lub {int(eval_score * 100)} centypionach) | Prawdziwa: {correct_eval}")

    # STATYSTYKA
    evaluate_model_metrics(model, val_loader, device)
