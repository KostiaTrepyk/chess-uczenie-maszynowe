import torch
import os
import shutil
from core.architecture import ChessResNet

def export_model_to_onnx():
    # Ustawienie procesora (CPU) dla eksportu
    device = torch.device('cpu')
    model = ChessResNet().to(device)
    
    # Wczytanie najlepszych wag modelu
    model.load_state_dict(torch.load("best_chess_model.pth", map_location=device, weights_only=True))
    
    model.eval()

    # Przykładowe wejście do wygenerowania grafu ONNX
    dummy_input = torch.randn(4, 15, 8, 8, device=device)

    # Dynamiczny rozmiar partii (pozwala na zmienny batch size)
    dynamic_axes = {
        'input': {0: 'batch'},
        'score': {0: 'batch'}, 
        'mate': {0: 'batch'}   
    }

    out_path = "chess_model.onnx"
    
    print("Trwa eksportowanie modelu do ONNX...")
    
    # Eksport do formatu ONNX
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

    # Przeniesienie plików do Next.js
    try:    
        target_dir = os.path.join(os.getcwd(), "..", "webapp", "models")
        
        os.makedirs(target_dir, exist_ok=True)
        
        # Kopiowanie głównego pliku modelu
        shutil.copy(out_path, os.path.join(target_dir, "chess_model.onnx"))
        
        # Kopiowanie pliku z wagami
        data_path = out_path + ".data"
        if os.path.exists(data_path):
            shutil.copy(data_path, os.path.join(target_dir, "chess_model.onnx.data"))
            
        print(f"✅ pomyślnie wyeksportowano i skopiowano do {target_dir}")
    except Exception as e:
        print("⚠️ Wyeksportowano ONNX, ale wystąpił błąd przy kopiowaniu:", e)
