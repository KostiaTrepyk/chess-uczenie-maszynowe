from core.dataset import load_dataset, prepare_dataset
from core.model import show_model_stats, train_model

if __name__ == '__main__':
    # Загрузка данных, обучение модели 
    df = load_dataset()

    train_loader, val_loader = prepare_dataset(df)

    model = train_model(train_loader, val_loader)

    # Оценка модели
    print('\nОценка модели на валидационном наборе данных:')
    df_test = load_dataset(100_000)
    _, val_loader_test = prepare_dataset(df_test)

    show_model_stats(model, val_loader_test, df_test)
