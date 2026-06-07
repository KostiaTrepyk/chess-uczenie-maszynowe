import torch
import torch.nn as nn

from core.ChessNNUE import ChessNNUE

def main():
    model_path = "best_nnue_model.pth" # Укажи точный путь к твоему файлу на 1.422
    onnx_path = "chess_nnue.onnx"

    print("Инициализация модели...")
    model = ChessNNUE()
    
    # Загружаем веса
    model.load_state_dict(torch.load(model_path, map_location='cpu', weights_only=True))
    model.eval()

    # Создаем "фиктивные" входные данные (размер батча 1, 32 фигуры максимум)
    dummy_w = torch.zeros(1, 32, dtype=torch.long)
    dummy_b = torch.zeros(1, 32, dtype=torch.long)
    dummy_state = torch.zeros(1, 6, dtype=torch.float32)

    print(f"Экспорт в {onnx_path}...")
    torch.onnx.export(
        model,
        (dummy_w, dummy_b, dummy_state),
        onnx_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=['w_indices', 'b_indices', 'state'],
        output_names=['score', 'mate'], # <--- ТЕПЕРЬ ДВА ВЫХОДА
        dynamic_axes={
            'w_indices': {0: 'batch_size', 1: 'num_pieces'},
            'b_indices': {0: 'batch_size', 1: 'num_pieces'},
            'state': {0: 'batch_size'},
            'score': {0: 'batch_size'},
            'mate': {0: 'batch_size'}
        }
    )
    
    print("Готово! Модель успешно экспортирована.")

if __name__ == '__main__':
    main()