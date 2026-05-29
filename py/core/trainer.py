import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from core.architecture import ChessResNet
from core.features import build_batch_on_gpu

def train_model(train_loader : DataLoader, val_loader : DataLoader) -> ChessResNet:
    device_str = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nTrenowanie na urządzeniu: {device}")

    # Włączamy akcelerację sprzętową dla sieci splotowych (Daje +10-15% do szybkości)
    torch.backends.cudnn.benchmark = True

    model = ChessResNet().to(device)
    
    # Kompilacja modelu dla PyTorch 2.0+ (Daje jeszcze +20% do szybkości). W razie błędu na Windowsie - po prostu usuń tę linię.
    # model = torch.compile(model) 

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()
    epochs = 100

    patience = 5
    best_val_loss = float('inf')
    epochs_no_improve = 0

    scaler = torch.GradScaler()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2
    )

    print("\nRozpoczynamy trenowanie...")
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        train_pbar = tqdm(train_loader, desc=f"Epoka {epoch+1}/{epochs} [Train]", leave=False)
        for batch_data in train_pbar:
            # 1. Przenosimy surowe i lekkie dane na kartę graficzną (GPU)
            boards, turns, castling, eps, batch_targets = [x.to(device, non_blocking=True) for x in batch_data]
            
            # 2. Błyskawicznie budujemy ciężkie macierze na GPU
            batch_inputs = build_batch_on_gpu(boards, turns, castling, eps)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(device_str):
                # PyTorch automatycznie zastosuje FlashAttention wewnątrz Transformera
                predictions = model(batch_inputs)
                loss = loss_fn(predictions, batch_targets)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()
            train_pbar.set_postfix({'loss': f"{loss.item():.4f}"})

        avg_train_loss = train_loss / len(train_loader)

        model.eval()
        val_loss = 0.0

        val_pbar = tqdm(val_loader, desc=f"Epoka {epoch+1}/{epochs} [Val]", leave=False)
        with torch.no_grad():
            for val_data in val_pbar:
                # To samo dla walidacji: przenosimy i budujemy
                boards, turns, castling, eps, val_targets = [x.to(device, non_blocking=True) for x in val_data]
                val_inputs = build_batch_on_gpu(boards, turns, castling, eps)

                val_preds = model(val_inputs)
                v_loss = loss_fn(val_preds, val_targets)
                val_loss += v_loss.item()
                
                val_pbar.set_postfix({'loss': f"{v_loss.item():.4f}"})

        avg_val_loss = val_loss / len(val_loader)

        # Zostawiamy tylko ostateczny wynik, paski znikną (leave=False)
        print(f"\tEpoka {epoch+1} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        scheduler.step(avg_val_loss)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), "best_chess_model.pth")
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print(f"\n🛑 WCZESNE ZATRZYMANIE (EARLY STOPPING)!")
            break

    model.load_state_dict(torch.load("best_chess_model.pth"))
    model.eval()
    print("Pomyślnie wczytano najlepszą wersję modelu.")
    
    return model
