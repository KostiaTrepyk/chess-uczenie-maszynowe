import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from core.architecture import ChessResNet
from core.features import build_batch_on_gpu
from core.consts import checks_per_epoch, max_val_batches

def train_model(train_loader : DataLoader, val_loader : DataLoader) -> ChessResNet:
    device_str = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nTrenowanie na urządzeniu: {device}")

    # Włączamy akcelerację sprzętową dla sieci splotowych
    torch.backends.cudnn.benchmark = True

    model = ChessResNet().to(device)
    # model = torch.compile(model) # Zostawione zakomentowane ze względu na Windowsa

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.05)
    epochs = 100

    patience = 5 * checks_per_epoch  
    best_val_loss = float('inf')
    epochs_no_improve = 0

    # 1. СНАЧАЛА считаем батчи
    total_batches = len(train_loader)
    val_interval = max(1, total_batches // checks_per_epoch)

    scaler = torch.GradScaler()
    
    # 2. ПОТОМ передаем их в OneCycleLR
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=0.001,
        epochs=epochs,
        steps_per_epoch=total_batches,
        pct_start=0.1,
        div_factor=10.0,
        final_div_factor=1000.0
    )

    print("\nRozpoczynamy trenowanie...")
    print(f"Całkowita liczba wsadów (batches) na epokę: {total_batches}")
    print(f"Walidacja i zapis co {val_interval} wsadów.")

    for epoch in range(epochs):
        model.train()
        running_train_loss = 0.0
        steps_since_val = 0

        train_pbar = tqdm(train_loader, desc=f"Epoka {epoch+1}/{epochs} [Train]", leave=False)
        for step, batch_data in enumerate(train_pbar):
            # 1. Przenosimy dane na GPU
            boards, turns, castling, eps, batch_targets = [x.to(device, non_blocking=True) for x in batch_data]
            
            # 2. Budujemy macierze
            batch_inputs = build_batch_on_gpu(boards, turns, castling, eps)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(device_str):
                predictions = model(batch_inputs)
                loss = loss_fn(predictions, batch_targets)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            scheduler.step()
            
            running_train_loss += loss.item()
            steps_since_val += 1
            
            # Объединенный вывод метрик в tqdm (без затирания)
            current_lr = scheduler.get_last_lr()[0]
            train_pbar.set_postfix({'loss': f"{loss.item():.4f}", 'lr': f"{current_lr:.6f}"})

            # --- WALIDACJA WEWNĄTRZ EPOKI ---
            if (step + 1) % val_interval == 0 or (step + 1) == total_batches:
                avg_train_loss = running_train_loss / steps_since_val
                
                model.eval()
                val_loss = 0.0

                val_pbar = tqdm(val_loader, desc=f"Epoka {epoch+1} [Val] Krok {step+1}", leave=False)
                
                val_steps_taken = 0
                
                with torch.no_grad():
                    for val_data in val_pbar:
                        if val_steps_taken >= max_val_batches:
                            break 
                            
                        boards, turns, castling, eps, val_targets = [x.to(device, non_blocking=True) for x in val_data]
                        val_inputs = build_batch_on_gpu(boards, turns, castling, eps)

                        val_preds = model(val_inputs)
                        v_loss = loss_fn(val_preds, val_targets)
                        val_loss += v_loss.item()
                        
                        val_steps_taken += 1
                        val_pbar.set_postfix({'loss': f"{v_loss.item():.4f}"})

                avg_val_loss = val_loss / val_steps_taken

                print(f"\n\t[Epoka {epoch+1} | Krok {step+1}/{total_batches}] Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    epochs_no_improve = 0
                    torch.save(model.state_dict(), "best_chess_model.pth")
                    print(f"\t>>> Zapisano nowy najlepszy model! (Loss: {best_val_loss:.4f})")
                else:
                    epochs_no_improve += 1

                if epochs_no_improve >= patience:
                    print(f"\n🛑 WCZESNE ZATRZYMANIE (EARLY STOPPING)!")
                    model.load_state_dict(torch.load("best_chess_model.pth"))
                    model.eval()
                    return model

                # Resetujemy statystyki i wracamy do treningu
                running_train_loss = 0.0
                steps_since_val = 0
                model.train() 

    # Po zakończeniu wszystkich epok wczytujemy najlepszy model
    model.load_state_dict(torch.load("best_chess_model.pth"))
    model.eval()
    print("\nPomyślnie wczytano najlepszą wersję modelu.")
    
    return model
