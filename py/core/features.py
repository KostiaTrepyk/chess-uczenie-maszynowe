import math

import numpy as np
import torch
import torch.nn.functional as F

from core.consts import CHAR_TO_INT_TOKEN, MIN_EVAL, MAX_EVAL, WDL_SCALE
    
def clean_eval_to_float(eval_str: str | float) -> float:
    """Конвертирует оценку в пешках в вероятность победы (0.0 ... 1.0)."""
    eval_str = str(eval_str).strip()
    if eval_str.startswith('\ufeff'): eval_str = eval_str[1:]
    
    if eval_str.startswith('#'):
        eval_pawns = 10.0 if eval_str[1] == '+' else -10.0
    else:
        eval_pawns = float(eval_str) / 100.0
        
    eval_pawns = max(MIN_EVAL, min(MAX_EVAL, eval_pawns))
    
    # Формула конвертации пешек в вероятность победы (Sigmoid)
    win_prob = 1.0 / (1.0 + math.exp(-eval_pawns / WDL_SCALE))
    return win_prob

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
    b = boards.size(0)
    device = boards.device
    
    # 1. Bierki (12 warstw)
    pieces = F.one_hot(boards, num_classes=13)[:, :, :12].float()
    pieces = pieces.permute(0, 2, 1).view(b, 12, 8, 8)
    
    # --- МАГИЯ КАНОНИЧЕСКОЙ ОРИЕНТАЦИИ ---
    black_turn = (turns == 0)
    
    if black_turn.any():
        # Отражаем доску по вертикали для черных (строки 1-8 меняются на 8-1)
        pieces[black_turn] = torch.flip(pieces[black_turn], dims=[2])
        
        # Меняем цвета фигур местами (белые 0-5 <-> черные 6-11)
        pieces_black = pieces[black_turn].clone()
        pieces_swapped = torch.cat([pieces_black[:, 6:12, :, :], pieces_black[:, 0:6, :, :]], dim=1)
        pieces[black_turn] = pieces_swapped
    # --------------------------------------

    # 2. Kolej ruchu (1 warstwa)
    turns_layer = torch.ones((b, 1, 8, 8), device=device, dtype=torch.float32)
    
    # 3. Roszada (1 warstwa)
    castling_layer = torch.zeros((b, 1, 8, 8), device=device, dtype=torch.float32)
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
    
    ep_black_mask = black_turn[has_ep]
    if ep_black_mask.any():
        ep_idx[ep_black_mask] = ep_idx[ep_black_mask] ^ 56
        
    ep_layer[has_ep, 0, ep_idx] = 1.0 
    ep_layer = ep_layer.view(b, 1, 8, 8)
    
    return torch.cat([pieces, turns_layer, castling_layer, ep_layer], dim=1)

def fen_to_tensor(fen: str) -> torch.Tensor:
    b, t, c, e = tokenize_fen(fen)
    boards = torch.tensor(b, dtype=torch.long).unsqueeze(0)
    turns = torch.tensor(t, dtype=torch.float32).unsqueeze(0)
    castling = torch.tensor(c, dtype=torch.float32).unsqueeze(0)
    eps = torch.tensor(e, dtype=torch.long).unsqueeze(0)
    return build_batch_on_gpu(boards, turns, castling, eps).squeeze(0)
