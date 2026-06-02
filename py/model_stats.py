from core.dataset import load_dataset, prepare_dataset_with_split
from core.architecture import load_model
from core.metrics import evaluate_model_metrics, show_model_stats

if __name__ == '__main__':
    model = load_model()

    df = load_dataset()
    _, val_loader = prepare_dataset_with_split(df)

    show_model_stats(model, val_loader, df)
