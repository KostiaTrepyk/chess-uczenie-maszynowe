from core.dataset import prepare_dataset, load_dataset
from core.architecture import load_model
from core.metrics import show_model_stats

if __name__ == '__main__':
    model = load_model()

    df = load_dataset(250_000)
    train_loader, val_loader = prepare_dataset(df)

    show_model_stats(model, val_loader, df)
