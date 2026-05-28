import pandas as pd
import math
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from tqdm import tqdm
from pandas import DataFrame
from torch.utils.data import DataLoader, random_split

class FastChessDataset(Dataset):
    def __init__(self, df):
        n = len(df)
        self.boards = np.zeros((n, 64), dtype=np.uint8)
        self.turns = np.zeros(n, dtype=np.uint8)
        self.castling = np.zeros((n, 4), dtype=np.uint8)
        self.ep = np.zeros(n, dtype=np.uint8) # <-- Массив для En Passant
        self.targets = np.zeros(n, dtype=np.float32)

        fens = df['FEN'].values
        evals = df['Evaluation'].values

        print("Загрузка данных в ОЗУ (это займет пару минут)...")
        for i in tqdm(range(n)):
            b, t, c, e = tokenize_fen(fens[i])
            self.boards[i] = b
            self.turns[i] = t
            self.castling[i] = c
            self.ep[i] = e # <-- Сохраняем
            self.targets[i] = clean_eval(evals[i])

    def __len__(self): return len(self.boards)

    def __getitem__(self, idx):
        board_idx = torch.tensor(self.boards[idx], dtype=torch.long)
        board_layers = F.one_hot(board_idx, num_classes=13)[:, :12].float()
        board_layers = board_layers.view(8, 8, 12).permute(2, 0, 1)

        # ТЕПЕРЬ ТЕНЗОР (15, 8, 8)
        tensor = torch.zeros((15, 8, 8), dtype=torch.float32)
        tensor[:12, :, :] = board_layers
        
        if self.turns[idx] == 1: tensor[12, :, :] = 1.0
        
        c = self.castling[idx]
        if c[0]: tensor[13, 7, 7] = 1.0
        if c[1]: tensor[13, 7, 0] = 1.0
        if c[2]: tensor[13, 0, 7] = 1.0
        if c[3]: tensor[13, 0, 0] = 1.0

        # Разворачиваем En Passant
        ep_val = self.ep[idx]
        if ep_val > 0:
            ep_idx = ep_val - 1
            ep_row = ep_idx // 8
            ep_col = ep_idx % 8
            tensor[14, ep_row, ep_col] = 1.0
            
        target = self.targets[idx] / 10.0 
        return tensor, torch.tensor([target], dtype=torch.float32)
    
def clean_eval(eval_str):
    eval_str = str(eval_str).strip()
    if eval_str.startswith('\ufeff'): eval_str = eval_str[1:]
    if eval_str.startswith('#'):
        return 10.0 if eval_str[1] == '+' else -10.0
    
    eval_pawns = float(eval_str) / 100.0
    return max(-10.0, min(10.0, eval_pawns)) 

# Словарь для токенизации (0-11 для фигур, 12 - для пустой клетки)
CHAR_TO_INT_TOKEN = {
    'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5,
    'p': 6, 'n': 7, 'b': 8, 'r': 9, 'q': 10, 'k': 11
}

def tokenize_fen(fen: str):
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
        
    # <-- НОВОЕ: Взятие на проходе
    ep = np.uint8(0) 
    if len(parts) > 3 and parts[3] != '-':
        col = ord(parts[3][0]) - ord('a')
        row = 8 - int(parts[3][1])
        # Сохраняем координату (плюс 1, чтобы 0 означало "нет")
        ep = np.uint8(row * 8 + col + 1)
        
    return board, turn, castling, ep

def prepare_dataset(df: DataFrame) -> tuple[DataFrame, DataLoader, DataLoader]:
    print('Подготовка датасета...')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset = FastChessDataset(df)

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_dataset,
        batch_size=4096,
        shuffle=True,
        num_workers=0,
        pin_memory=True if device == 'cuda' else False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=4096,
        shuffle=False,
        num_workers=0,
        pin_memory=True if device == 'cuda' else False
    )
    
    print('Подготовка датасета завершена.')
    return (train_loader, val_loader)

def load_dataset(n : int | None) -> DataFrame:
    print('Загрузка данных...')
    df = pd.read_csv("chessData.csv").sample(n=n, random_state=42)
    print('Загрузка данных завершена.')
    return df

