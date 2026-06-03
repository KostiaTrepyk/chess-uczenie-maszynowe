import torch
from core.architecture import ChessResNet

def export_model_to_onnx():
    device = torch.device('cpu')
    model = ChessResNet().to(device)
    model.load_state_dict(torch.load("best_chess_model.pth", map_location=device, weights_only=True))
    model.eval()

    # СТРОГО 1 ДОСКА: убираем динамику, чтобы Трансформер работал идеально
    dummy_input = torch.randn(1, 15, 8, 8, device=device)

    torch.onnx.export(
        model,
        dummy_input,
        "chess_model.onnx",
        export_params=True,
        opset_version=18,
        input_names=['input'],
        output_names=['output']
        # dynamic_axes УДАЛЕНЫ
    )
    print("✅ Plik chess_model.onnx jest gotowy (Strict Batch = 1)!")
