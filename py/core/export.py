import torch

from core.architecture import ChessResNet

def export_model_to_onnx():
    device = torch.device('cpu')
    
    # 1. Инициализируем модель без аргументов (она сама возьмет 256 каналов и 20 блоков)
    model = ChessResNet().to(device)
    
    # 2. Загружаем веса (добавлен weights_only=True для безопасности)
    model.load_state_dict(torch.load("best_chess_model.pth", map_location=device, weights_only=True))
    model.eval()

    # 3. Создаем макет входа [Batch, Channels, Height, Width]
    dummy_input = torch.randn(1, 15, 8, 8, device=device)

    # 4. Экспортируем в ONNX
    torch.onnx.export(
        model,
        dummy_input,
        "chess_model.onnx",
        export_params=True,
        opset_version=18,
        input_names=['input'],
        output_names=['output'],
        # КРИТИЧЕСКИ ВАЖНО: Делаем размер батча (индекс 0) динамическим
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
    print("✅ Plik chess_model.onnx jest gotowy!")