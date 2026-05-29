import numpy as np
import pandas as pd
from pandas import DataFrame
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm

from core.features import tokenize_fen, clean_eval_to_class

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

        print("Ładowanie danych do pamięci RAM (to zajmie kilka minut)...")
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
        # Maksymalnie szybkie odczytanie surowych danych
        board = torch.tensor(self.boards[idx], dtype=torch.long)
        turn = torch.tensor(self.turns[idx], dtype=torch.float32)
        castling = torch.tensor(self.castling[idx], dtype=torch.float32)
        ep = torch.tensor(self.ep[idx], dtype=torch.long)
        
        target_class = int(self.targets[idx]) 
        
        return board, turn, castling, ep, torch.tensor(target_class, dtype=torch.long)

def prepare_dataset(df: DataFrame) -> tuple[DataLoader, DataLoader]:
    print('\nPrzygotowanie zestawu danych...')
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
    
    print('Przygotowanie zestawu danych zakończone.')
    return train_loader, val_loader

def load_dataset(n: int | None = None) -> DataFrame:
    print('\nŁadowanie danych...')
    df = pd.read_csv("chessData.csv").sample(n=n, random_state=42) if n is not None else pd.read_csv("chessData.csv")
    print('Ładowanie danych zakończone.')
    return df
