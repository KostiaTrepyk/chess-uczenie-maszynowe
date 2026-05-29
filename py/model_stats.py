import torch

from core.dataset import prepare_dataset, load_dataset
from core.architecture import load_model
from core.metrics import evaluate_model_metrics, show_model_stats

if __name__ == '__main__':
    model = load_model()

    df = load_dataset(50_000)
    loader = prepare_dataset(df)

    # show_model_stats(model, loader, df)
    evaluate_model_metrics(model, loader, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
