import torch
from core.architecture import ChessResNet

def export_model_to_onnx():
    device = torch.device('cpu')
    model = ChessResNet().to(device)
    model.load_state_dict(torch.load("best_chess_model.pth", map_location=device, weights_only=True))
    model.eval()

    # Используем динамический batch (первое измерение), чтобы ONNX поддерживал батчинг
    # Use a small batch >1 so ONNX export records batched ops correctly
    dummy_input = torch.randn(4, 15, 8, 8, device=device)

    dynamic_axes = {
        'input': {0: 'batch'},
        'output': {0: 'batch'}
    }

    out_path = "chess_model.onnx"
    torch.onnx.export(
        model,
        dummy_input,
        out_path,
        export_params=True,
        opset_version=18,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes=dynamic_axes
    )

    # Also copy the exported ONNX to the webapp models folder so the Next server can load it
    try:
        import shutil, os
        target_dir = os.path.join(os.getcwd(), "webapp", "models")
        os.makedirs(target_dir, exist_ok=True)
        shutil.copy(out_path, os.path.join(target_dir, "chess_model.onnx"))
        print(f"✅ chess_model.onnx exported and copied to {target_dir}")
    except Exception as e:
        print("⚠️ Exported ONNX but failed to copy to webapp/models:", e)
