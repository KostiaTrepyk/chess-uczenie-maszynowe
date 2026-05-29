import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm
from pandas import DataFrame

# Словарь для токенизации (0-11 для фигур, 12 - для пустой клетки)
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

class FastChessDataset(Dataset):
    def __init__(self, df: DataFrame):
        n = len(df)
        self.boards = np.zeros((n, 64), dtype=np.uint8)
        self.turns = np.zeros(n, dtype=np.uint8)
        self.castling = np.zeros((n, 4), dtype=np.uint8)
        self.ep = np.zeros(n, dtype=np.uint8) 
        self.targets = np.zeros(n, dtype=np.int64)

        fens = df['FEN'].values
        evals = df['Evaluation'].values

        print("Загрузка данных в ОЗУ (это займет пару минут)...")
        for i in tqdm(range(n)):
            b, t, c, e = tokenize_fen(fens[i])
            self.boards[i] = b
            self.turns[i] = t
            self.castling[i] = c
            self.ep[i] = e 
            self.targets[i] = clean_eval_to_class(evals[i])

    def __len__(self) -> int: 
        return len(self.boards)

    def __getitem__(self, idx: int) -> tuple:
        # Максимально быстрый сброс сырых данных
        board = torch.tensor(self.boards[idx], dtype=torch.long)
        turn = torch.tensor(self.turns[idx], dtype=torch.float32)
        castling = torch.tensor(self.castling[idx], dtype=torch.float32)
        ep = torch.tensor(self.ep[idx], dtype=torch.long)
        
        target_class = int(self.targets[idx]) 
        
        return board, turn, castling, ep, torch.tensor(target_class, dtype=torch.long)

def prepare_dataset(df: DataFrame) -> tuple[DataLoader, DataLoader]:
    print('\nПодготовка датасета...')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    is_cuda = (device.type == 'cuda')

    dataset = FastChessDataset(df)

    train_size = int(0.9 * len(dataset)) # Обычно 90/10 лучше для больших датасетов
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_dataset,
        batch_size=1024,
        shuffle=True,
        num_workers=0,
        pin_memory=is_cuda,
        persistent_workers=False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=1024,
        shuffle=False,
        num_workers=0,
        pin_memory=is_cuda,
        persistent_workers=False
    )
    
    print('Подготовка датасета завершена.')
    return train_loader, val_loader

def load_dataset(n: int | None = None) -> DataFrame:
    print('\nЗагрузка данных...')
    df = pd.read_csv("chessData.csv").sample(n=n, random_state=42) if n is not None else pd.read_csv("chessData.csv")
    print('Загрузка данных завершена.')
    return df