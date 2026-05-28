from core.model import load_model, show_model_stats
from core.dataset import prepare_dataset, load_dataset

model = load_model()

df = load_dataset(100_000)
train_loader, val_loader = prepare_dataset(df)

show_model_stats(model, val_loader, df)