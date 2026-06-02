import os
import numpy as np
import pandas as pd
from pandas import DataFrame
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm

from core.features import clean_eval_to_float, tokenize_fen
from core.consts import BATCH_SIZE, MAX_VALIDATION_BATCHES

class FastChessDataset(Dataset):
    def __init__(self, df: DataFrame, cache_dir: str = "cache"):
        n = len(df)
        os.makedirs(cache_dir, exist_ok=True)
        # Уникальное имя кеша на основе количества строк
        cache_file = os.path.join(cache_dir, f"fastchess_cache_{n}.pt")

        if os.path.exists(cache_file):
            print(f"Wczytywanie zbuforowanego zestawu danych z {cache_file} (błyskawicznie)...")
            cached_data = torch.load(cache_file, weights_only=False)
            self.boards = cached_data['boards']
            self.turns = cached_data['turns']
            self.castling = cached_data['castling']
            self.ep = cached_data['ep']
            self.targets = cached_data['targets']
        else:
            print("Ładowanie danych do pamięci RAM (to zajmie kilka minut)...")
            self.boards = np.zeros((n, 64), dtype=np.uint8)
            self.turns = np.zeros(n, dtype=np.uint8)
            self.castling = np.zeros((n, 4), dtype=np.uint8)
            self.ep = np.zeros(n, dtype=np.uint8) 
            self.targets = np.zeros(n, dtype=np.float32)

            fens = df['FEN'].values
            evals = df['Evaluation'].values

            for i in tqdm(range(n)):
                b, t, c, e = tokenize_fen(fens[i])
                self.boards[i] = b
                self.turns[i] = t
                self.castling[i] = c
                self.ep[i] = e 
                self.targets[i] = clean_eval_to_float(evals[i])
            
            print(f"Zapisywanie przetworzonych danych do {cache_file}...")
            torch.save({
                'boards': self.boards,
                'turns': self.turns,
                'castling': self.castling,
                'ep': self.ep,
                'targets': self.targets
            }, cache_file)

    def __len__(self) -> int: 
        return len(self.boards)

    def __getitem__(self, idx: int) -> tuple:
        board = torch.tensor(self.boards[idx], dtype=torch.long)
        turn = torch.tensor(self.turns[idx], dtype=torch.float32)
        castling = torch.tensor(self.castling[idx], dtype=torch.float32)
        ep = torch.tensor(self.ep[idx], dtype=torch.long)
        target_val = torch.tensor(self.targets[idx], dtype=torch.float32)
        
        return board, turn, castling, ep, target_val

def prepare_dataset_with_split(df: DataFrame) -> tuple[DataLoader, DataLoader]:
    print('\nPrzygotowanie zestawu danych...')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    is_cuda = (device.type == 'cuda')

    dataset = FastChessDataset(df)

    # ИСПРАВЛЕНИЕ: Статичный размер валидации (например, 102,400 позиций)
    val_size = MAX_VALIDATION_BATCHES * BATCH_SIZE 
    # Все остальное (почти 12.9 млн) идет в тренировку
    train_size = len(dataset) - val_size 

    # Фиксируем seed, чтобы валидационный сет всегда состоял из одних и тех же позиций
    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=generator)

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, 
        num_workers=0, pin_memory=is_cuda, persistent_workers=False
    )

    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False, 
        num_workers=0, pin_memory=is_cuda, persistent_workers=False
    )
    
    print('Przygotowanie zestawu danych zakończone.')
    return train_loader, val_loader

def prepare_dataset(df: DataFrame) -> DataLoader:
    print('\nPrzygotowanie zestawu danych...')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    is_cuda = (device.type == 'cuda')

    dataset = FastChessDataset(df)

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=is_cuda,
        persistent_workers=False
    )
    
    print('Przygotowanie zestawu danych zakończone.')
    return loader

def load_dataset(n: int | None = None) -> DataFrame:
    print('\nŁadowanie danych...')
    df = pd.read_csv("chessData.csv").sample(n=n, random_state=42) if n is not None else pd.read_csv("chessData.csv")
    print('Ładowanie danych zakończone.')
    return df
