import torch
import os
from torch.utils.data import DataLoader, random_split

from core.ChessNNUE import ChessNNUE, NNUEChessDataset, evaluate_nnue

def main():
    model_path = "best_nnue_model.pth"
    data_file_path = "chessData.csv" # Укажи реальный путь к датасету

    if not os.path.exists(model_path):
        print(f"BŁĄD: Nie znaleziono pliku modelu: {model_path}")
        return

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Urządzenie: {device}")

    # 1. Загрузка данных и точное воссоздание валидационного набора
    print(f"Ładowanie danych z {data_file_path}...")
    full_dataset = NNUEChessDataset(data_file_path)
    
    if len(full_dataset) == 0:
        return

    total_size = len(full_dataset)
    train_size = int(0.9 * total_size)
    val_size = total_size - train_size
    
    # Использование manual_seed(42) обязательно, чтобы не смешать Train и Val!
    _, val_dataset = random_split(
        full_dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=16384,
        shuffle=False,
        num_workers=4, # Измени на 0, если тестируешь на Windows без __main__ защиты
        pin_memory=True
    )

    # 2. Инициализация архитектуры и безопасная загрузка весов
    print("Inicjalizacja architektury i ładowanie wag...")
    model = ChessNNUE().to(device)
    
    # weights_only=True — стандарт безопасности PyTorch
    # map_location=device — защищает от краша, если модель обучалась на GPU, а тестируется на CPU
    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)

    # 3. Запуск оценки
    evaluate_nnue(model, val_loader, device)

if __name__ == '__main__':
    main()