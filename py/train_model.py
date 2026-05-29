from core.dataset import load_dataset, prepare_dataset
from core.trainer import train_model
from core.metrics import show_model_stats

if __name__ == '__main__':
    # Wczytywanie danych, trenowanie modelu 
    df = load_dataset(5_000_000)

    train_loader, val_loader = prepare_dataset(df)

    model = train_model(train_loader, val_loader)

    # Ocena modelu
    print('\nOcena modelu na walidacyjnym zbiorze danych:')
    df_test = load_dataset(100_000)
    _, val_loader_test = prepare_dataset(df_test)

    show_model_stats(model, val_loader_test, df_test)
