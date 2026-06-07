import torch
from core.architecture import ChessResNet

def export_model_to_onnx():
    device = torch.device('cpu')
    model = ChessResNet().to(device)
    model.load_state_dict(torch.load("best_chess_model.pth", map_location=device, weights_only=True))
    model.eval()

    # Используем динамический batch (первое измерение)
    dummy_input = torch.randn(4, 15, 8, 8, device=device)

    dynamic_axes = {
        'input': {0: 'batch'},
        'score': {0: 'batch'}, 
        'mate': {0: 'batch'}   
    }

    out_path = "chess_model.onnx"
    
    print("Trwa eksportowanie modelu do ONNX...")
    # Оставляем только ОДИН правильный вызов экспорта
    torch.onnx.export(
        model,
        dummy_input,
        out_path,
        export_params=True,
        opset_version=18,
        input_names=['input'],
        output_names=['score', 'mate'], 
        dynamic_axes=dynamic_axes
    )

    # Копируем в папку Next.js
    try:
        import shutil, os
        target_dir = os.path.join(os.getcwd(), "..", "webapp", "models")
        os.makedirs(target_dir, exist_ok=True)
        shutil.copy(out_path, os.path.join(target_dir, "chess_model.onnx"))
        shutil.copy(out_path + ".data", os.path.join(target_dir, "chess_model.onnx.data"))
        print(f"✅ chess_model.onnx i chess_model.onnx.data wyeksportowano i skopiowano do {target_dir}")
    except Exception as e:
        print("⚠️ Wyeksportowano ONNX, ale nie udało się skopiować do webapp/models:", e)