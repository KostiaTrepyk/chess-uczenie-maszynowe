import torch

from core.architecture import ChessResNet
from core.consts import NUM_BLOCKS, CHANNELS

def export_model_to_onnx(  num_blocks=NUM_BLOCKS, channels=CHANNELS):
    device = torch.device('cpu')
    model = ChessResNet(channels=channels, num_blocks=num_blocks).to(device)
    model.load_state_dict(torch.load("best_chess_model.pth", map_location=device))
    model.eval()

    # Tworzymy fałszywy tensor o odpowiednim kształcie (1 batch, 15 warstw, 8x8)
    dummy_input = torch.randn(1, 15, 8, 8, device=device)

    # Eksportujemy
    torch.onnx.export(
        model,
        dummy_input,
        "chess_model.onnx",
        export_params=True,
        opset_version=18,
        input_names=['input'],
        output_names=['output']
    )
    print("✅ Plik chess_model.onnx jest gotowy!")
