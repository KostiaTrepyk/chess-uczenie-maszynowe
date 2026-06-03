from torch import device
import torch

from core.dataset import load_dataset, prepare_dataset_with_split
from core.trainer import train_model
from core.metrics import evaluate_model_metrics

if __name__ == '__main__':
    # Wczytywanie danych, trenowanie modelu 
    df = load_dataset() 

    train_loader, val_loader = prepare_dataset_with_split(df)

    model = train_model(train_loader, val_loader, resume_path="best_chess_model.pth")

    # Ocena modelu
    print('\nOcena modelu na walidacyjnym zbiorze danych:')
    df_test = load_dataset(100_000)
    _, val_loader_test = prepare_dataset_with_split(df_test)

    evaluate_model_metrics(model, val_loader_test, device('cuda' if torch.cuda.is_available() else 'cpu'))
