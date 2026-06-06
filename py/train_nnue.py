import torch
from torch.utils.data import random_split
from core.ChessNNUE import NNUEChessDataset, train_nnue

class FastTensorDataLoader:
    def __init__(self, subset, batch_size, shuffle=True):
        self.dataset = subset.dataset
        self.subset_indices = torch.tensor(subset.indices, dtype=torch.long)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.n_samples = len(self.subset_indices)

    def __iter__(self):
        if self.shuffle:
            rand_idx = torch.randperm(self.n_samples)
            self.active_indices = self.subset_indices[rand_idx]
        else:
            self.active_indices = self.subset_indices
        self.i = 0
        return self

    def __next__(self):
        if self.i >= self.n_samples:
            raise StopIteration
        
        batch_idx = self.active_indices[self.i : self.i + self.batch_size]
        
        # Теперь вытаскиваем 4 тензора, включая состояния
        w = self.dataset.w_indices[batch_idx].to(torch.long)
        b = self.dataset.b_indices[batch_idx].to(torch.long)
        states = self.dataset.states[batch_idx]
        targets = self.dataset.targets[batch_idx]
        
        self.i += self.batch_size
        return w, b, states, targets

    def __len__(self):
        return (self.n_samples + self.batch_size - 1) // self.batch_size

def main():
    data_file_path = "./chessData.csv" 
    
    print(f"Ładowanie danych z {data_file_path}...")
    full_dataset = NNUEChessDataset(data_file_path)
    
    if len(full_dataset) == 0:
        print("BŁĄD: Dataset jest pusty.")
        return

    total_size = len(full_dataset)
    train_size = int(0.9 * total_size)
    val_size = total_size - train_size
    
    train_dataset, val_dataset = random_split(
        full_dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    print(f"Dane załadowane: {train_size} treningowych, {val_size} walidacyjnych.")

    train_loader = FastTensorDataLoader(train_dataset, batch_size=4096*16, shuffle=True)
    val_loader = FastTensorDataLoader(val_dataset, batch_size=4096*16, shuffle=False)

    print("Inicjalizacja modelu i rozpoczęcie treningu...")
    trained_model = train_nnue(train_loader, val_loader)
    
    print("\nTrening zakończony sukcesem!")

if __name__ == '__main__':
    main()
