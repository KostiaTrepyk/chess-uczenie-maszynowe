import numpy as np
import torch
import torch.nn.functional as F

PIECE_TO_CHANNEL = {
    'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5,
    'p': 6, 'n': 7, 'b': 8, 'r': 9, 'q': 10, 'k': 11
}

CHAR_TO_INT_TOKEN = {
    'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5,
    'p': 6, 'n': 7, 'b': 8, 'r': 9, 'q': 10, 'k': 11
}

NUM_CLASSES = 100
MIN_EVAL = -10.0
MAX_EVAL = 10.0

def clean_eval_to_class(eval_str: str | float) -> int:
    """Конвертирует оценку в пешках в индекс класса [0 ... 99]"""
    eval_str = str(eval_str).strip()
    if eval_str.startswith('\ufeff'): eval_str = eval_str[1:]
    
    if eval_str.startswith('#'):
        eval_pawns = 10.0 if eval_str[1] == '+' else -10.0
    else:
        eval_pawns = float(eval_str) / 100.0
        
    # Жестко обрезаем диапазон
    eval_pawns = max(MIN_EVAL, min(MAX_EVAL, eval_pawns))
    
    # Нормализуем в диапазон [0, 1]
    normalized = (eval_pawns - MIN_EVAL) / (MAX_EVAL - MIN_EVAL)
    
    # Переводим в индекс класса (защита от выхода за пределы массива)
    class_idx = int(normalized * (NUM_CLASSES - 1))
    return min(NUM_CLASSES - 1, max(0, class_idx))

def tokenize_fen(fen: str) -> tuple[np.ndarray, np.uint8, np.ndarray, np.uint8]:
    """Разбирает FEN строку на числовые массивы."""
    parts = fen.split()
    board = np.full(64, 12, dtype=np.uint8)
    idx = 0
    for char in parts[0]:
        if char == '/': continue
        elif char.isdigit(): idx += int(char)
        else:
            board[idx] = CHAR_TO_INT_TOKEN[char]
            idx += 1
    
    turn = np.uint8(1) if len(parts) > 1 and parts[1] == 'w' else np.uint8(0)
    
    castling = np.zeros(4, dtype=np.uint8)
    if len(parts) > 2:
        if 'K' in parts[2]: castling[0] = 1
        if 'Q' in parts[2]: castling[1] = 1
        if 'k' in parts[2]: castling[2] = 1
        if 'q' in parts[2]: castling[3] = 1
        
    # Взятие на проходе
    ep = np.uint8(0) 
    if len(parts) > 3 and parts[3] != '-':
        col = ord(parts[3][0]) - ord('a')
        row = 8 - int(parts[3][1])
        ep = np.uint8(row * 8 + col + 1)
        
    return board, turn, castling, ep

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
