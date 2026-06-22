import numpy as np
import torch
import torch.nn.functional as F

from core.consts import CHAR_TO_INT_TOKEN
    
def clean_eval_to_float(eval_str: str | float) -> float:
    """Zwraca SUROWĄ ocenę w pionkach. Wymuszone maty to +/- 30.0"""
    eval_str = str(eval_str).strip()
    if eval_str.startswith('\ufeff'): eval_str = eval_str[1:]
    
    if eval_str.startswith('#'):
        # Wzmacniamy wagę mata do 30 pionów, aby sieć wyraźnie widziała różnicę
        eval_pawns = 30.0 if eval_str[1] == '+' else -30.0
    else:
        # Konwersja z centypionów na piony (np. 150 -> 1.5)
        eval_pawns = float(eval_str) / 100.0
        
    return eval_pawns

def tokenize_fen(fen: str) -> tuple[np.ndarray, np.uint8, np.ndarray, np.uint8]:
    """Rozbija tekstowy FEN na podstawowe tablice liczbowe."""
    parts = fen.split()
    board = np.full(64, 12, dtype=np.uint8) # 12 to kod pustego pola
    idx = 0
    for char in parts[0]:
        if char == '/': continue
        elif char.isdigit(): idx += int(char)
        else:
            board[idx] = CHAR_TO_INT_TOKEN[char]
            idx += 1
    
    # 1 jeśli ruch białych, 0 jeśli czarnych
    turn = np.uint8(1) if len(parts) > 1 and parts[1] == 'w' else np.uint8(0)
    
    # Prawa do roszady [biała krótka, biała długa, czarna krótka, czarna długa]
    castling = np.zeros(4, dtype=np.uint8)
    if len(parts) > 2:
        if 'K' in parts[2]: castling[0] = 1
        if 'Q' in parts[2]: castling[1] = 1
        if 'k' in parts[2]: castling[2] = 1
        if 'q' in parts[2]: castling[3] = 1
        
    # En Passant (bicie w przelocie) - zapisane jako indeks na planszy 1D
    ep = np.uint8(0) 
    if len(parts) > 3 and parts[3] != '-':
        col = ord(parts[3][0]) - ord('a')
        row = 8 - int(parts[3][1])
        ep = np.uint8(row * 8 + col + 1)
        
    return board, turn, castling, ep

def build_batch_on_gpu(boards, turns, castling, ep):
    """Tworzy gotowy, 15-kanałowy tensor 3D prosto w pamięci GPU."""
    b = boards.size(0)
    device = boards.device
    
    # 1. Figury (12 warstw, One-Hot Encoding)
    pieces = F.one_hot(boards, num_classes=13)[:, :, :12].float()
    pieces = pieces.permute(0, 2, 1).view(b, 12, 8, 8)
    
    # --- MAGIA ORIENTACJI KANONICZNEJ ---
    # Sieć zawsze "patrzy" na szachownicę z perspektywy gracza wykonującego ruch
    black_turn = (turns == 0)
    
    if black_turn.any():
        # Obracamy planszę pionowo (wiersze 1-8 stają się 8-1)
        pieces[black_turn] = torch.flip(pieces[black_turn], dims=[2])
        
        # Zamieniamy kolory figur miejscami (kanały 0-5 <-> 6-11)
        # Dzięki temu sieć uczy się atakować w jedną stronę, niezależnie czy gra białymi, czy czarnymi
        pieces_black = pieces[black_turn].clone()
        pieces_swapped = torch.cat([pieces_black[:, 6:12, :, :], pieces_black[:, 0:6, :, :]], dim=1)
        pieces[black_turn] = pieces_swapped
    # --------------------------------------

    # 2. Kolej ruchu (1 warstwa w całości wypełniona jedynkami, bo zawsze oceniamy "z naszej perspektywy")
    turns_layer = torch.ones((b, 1, 8, 8), device=device, dtype=torch.float32)
    
    # 3. Roszady (1 warstwa kodująca prawa w narożnikach)
    castling_layer = torch.zeros((b, 1, 8, 8), device=device, dtype=torch.float32)
    # Rozróżniamy roszady na "nasze" i "wroga" zależnie od tego, kto gra
    our_short = torch.where(black_turn, castling[:, 2], castling[:, 0])
    our_long  = torch.where(black_turn, castling[:, 3], castling[:, 1])
    opp_short = torch.where(black_turn, castling[:, 0], castling[:, 2])
    opp_long  = torch.where(black_turn, castling[:, 1], castling[:, 3])
    
    castling_layer[:, 0, 7, 7] = our_short
    castling_layer[:, 0, 7, 0] = our_long
    castling_layer[:, 0, 0, 7] = opp_short
    castling_layer[:, 0, 0, 0] = opp_long
    
    # 4. Bicie w przelocie - En Passant (1 warstwa)
    ep_layer = torch.zeros((b, 1, 64), device=device, dtype=torch.float32)
    has_ep = ep > 0 
    ep_idx = ep[has_ep] - 1 
    
    # Jeśli ruch czarnych, lustrujemy też cel en passant
    ep_black_mask = black_turn[has_ep]
    if ep_black_mask.any():
        ep_idx[ep_black_mask] = ep_idx[ep_black_mask] ^ 56
        
    ep_layer[has_ep, 0, ep_idx] = 1.0 
    ep_layer = ep_layer.view(b, 1, 8, 8)
    
    # Łączymy wszystkie 15 warstw w jeden wielki tensor przygotowany dla ResNet
    return torch.cat([pieces, turns_layer, castling_layer, ep_layer], dim=1)

def fen_to_tensor(fen: str) -> torch.Tensor:
    """Konwersja pojedynczego FENa na tensor (używane głównie podczas gry w Next.js/ONNX)"""
    b, t, c, e = tokenize_fen(fen)
    boards = torch.tensor(b, dtype=torch.long).unsqueeze(0)
    turns = torch.tensor(t, dtype=torch.float32).unsqueeze(0)
    castling = torch.tensor(c, dtype=torch.float32).unsqueeze(0)
    eps = torch.tensor(e, dtype=torch.long).unsqueeze(0)
    return build_batch_on_gpu(boards, turns, castling, eps).squeeze(0)
