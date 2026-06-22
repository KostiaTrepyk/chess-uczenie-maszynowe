import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from core.metrics import evaluate_model_metrics
from core.architecture import ChessResNet
from core.features import build_batch_on_gpu
from core.consts import CHECKS_PER_EPOCH, MAX_VALIDATION_BATCHES, EPOCHS, START_LR, RESUME_START_LR

def train_model(train_loader: DataLoader, val_loader: DataLoader, resume_path: str = None) -> ChessResNet:
    device_str = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device_str)
    print(f"\nTrenowanie na urządzeniu: {device}")

    # Optymalizacja pod GPU, przyspiesza operacje splotowe (conv2d)
    torch.backends.cudnn.benchmark = True
    model = ChessResNet().to(device)

    # Definicja funkcji błędu (Loss) dla różnych aspektów gry
    loss_wdl_fn = nn.MSELoss(reduction='none') # WDL: Win/Draw/Loss (Skalowane do 0..1)
    loss_pawns_fn = nn.SmoothL1Loss(beta=1.0, reduction='none') # Surowa różnica w pionkach
    loss_mate_fn = nn.BCEWithLogitsLoss(reduction='none') # Prawdopodobieństwo mata
    
    alpha = 0.05 # Waga błędu dla "matowej głowy"
    best_val_loss = float('inf')
    start_lr = START_LR
    start_epoch = 0

    # Optymalizator ze wsparciem regularyzacji (zapobiega przeuczeniu)
    optimizer = torch.optim.AdamW(model.parameters(), lr=start_lr, weight_decay=1e-4)
    total_batches = len(train_loader)
    
    # Scheduler: Zmienia Learning Rate w czasie (tu: start od niskiego, pik, powolny spadek)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=start_lr,
        steps_per_epoch=total_batches,
        epochs=EPOCHS,
        pct_start=0.1,
        div_factor=10.0
    )

    # 1. LOGIKA WZNOWIENIA TRENINGU LUB DOUCZANIA (FINE-TUNING)
    if resume_path:
        print(f"Wczytywanie z {resume_path}...")
        try:
            checkpoint = torch.load(resume_path, map_location=device, weights_only=False)
        except Exception:
            checkpoint = torch.load(resume_path, map_location=device)
        
        # Jeśli plik zawiera stany optymalizatora, to jest to przerwany trening
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            print("Znaleziono pełny checkpoint! Wznawiam trening dokładnie od miejsca przerwania.")
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        else:
            # Jeśli podano same wagi, to zaczynamy powolne douczanie
            print("Znaleziono tylko wagi. Rozpoczynam delikatne douczanie (Fine-Tuning) z CosineAnnealingLR.")
            model.load_state_dict(checkpoint)
            
            # Wymuszamy niski Learning Rate, aby nie zniszczyć dobrego modelu
            for param_group in optimizer.param_groups:
                param_group['lr'] = RESUME_START_LR
            
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=total_batches * EPOCHS,
                eta_min=1e-6
            )

        print("Ocena początkowego modelu (Baseline Validation)...")
        model.eval()
        val_loss = 0.0
        total_baseline_mae = 0.0  
        val_steps_taken = 0
        total_baseline_pos = 0
        
        # Wstępne sprawdzenie, jak dobrze gra wczytany model
        with torch.no_grad():
            for val_data in tqdm(val_loader, desc="Baseline Eval", leave=False):
                if val_steps_taken >= MAX_VALIDATION_BATCHES: break 
                boards, turns, castling, eps, val_targets = [x.to(device, non_blocking=True) for x in val_data]
                val_inputs = build_batch_on_gpu(boards, turns, castling, eps)
                
                # Zawsze oceniamy pozycję "od strony wykonującego ruch", dlatego negujemy
                val_targets = torch.where(turns == 0, -val_targets, val_targets)
        
                v_score_preds, v_mate_preds = model(val_inputs)
                
                # --- HYBRYDOWY BŁĄD (HYBRID LOSS) ---
                v_weights = torch.ones_like(val_targets)
                # Hard Example Mining: 4x większa kara za kardynalne błędy (>8 pionów na minusie)
                v_weights[torch.abs(val_targets) >= 8.0] = 4.0
                
                v_preds_wdl = torch.sigmoid(v_score_preds / 4.0)
                v_targets_wdl = torch.sigmoid(val_targets / 4.0)
                v_loss_wdl = (loss_wdl_fn(v_preds_wdl, v_targets_wdl) * v_weights).mean()
                
                v_loss_pawns = (loss_pawns_fn(v_score_preds, val_targets) * v_weights).mean()
                
                v_targets_mate = (torch.abs(val_targets) > 15.0).float()
                v_loss_mate = (loss_mate_fn(v_mate_preds, v_targets_mate) * v_weights).mean()
                
                # Połączony błąd z różnych metryk
                v_loss = (0.8 * v_loss_wdl) + (0.2 * v_loss_pawns) + (alpha * v_loss_mate)
                val_loss += v_loss.item()
                
                total_baseline_mae += torch.abs(v_score_preds - val_targets).sum().item()
                total_baseline_pos += val_targets.size(0)
                
                val_steps_taken += 1
        
        best_val_loss = val_loss / val_steps_taken
        best_val_mae = total_baseline_mae / total_baseline_pos
        
        print(f">>> Baseline Val Loss: {best_val_loss:.4f} | Val MAE: {best_val_mae:.2f}\n")

    # Early stopping (Zabezpieczenie przed stagnacją)
    patience = 5 * CHECKS_PER_EPOCH
    epochs_no_improve = 0
    val_interval = max(1, total_batches // CHECKS_PER_EPOCH)
    
    # Skaler przyspieszający obliczenia na kompatybilnych kartach graficznych (Mixed Precision)
    scaler = torch.GradScaler()

    print("\nRozpoczynamy trenowanie...")
    print(f"Całkowita liczba wsadów (batches) na epokę: {total_batches}")
    print(f"Walidacja i zapis co {val_interval} wsadów.")

    for epoch in range(start_epoch, EPOCHS):
        model.train()
        running_train_loss = 0.0
        steps_since_val = 0

        train_pbar = tqdm(train_loader, desc=f"Epoka {epoch+1}/{EPOCHS} [Train]", leave=False)
        for step, batch_data in enumerate(train_pbar):
            boards, turns, castling, eps, batch_targets = [x.to(device, non_blocking=True) for x in batch_data]
            batch_inputs = build_batch_on_gpu(boards, turns, castling, eps)
            
            # Odwrócenie metryki dla czarnych
            batch_targets = torch.where(turns == 0, -batch_targets, batch_targets)

            optimizer.zero_grad(set_to_none=True)

            # amp.autocast oblicza część sieci w 16-bitach (dużo szybciej i oszczędniej w VRAM)
            with torch.amp.autocast(device_str):
                score_preds, mate_preds = model(batch_inputs)
                
                weights = torch.ones_like(batch_targets)
                weights[torch.abs(batch_targets) >= 8.0] = 4.0 
                
                preds_wdl = torch.sigmoid(score_preds / 4.0)
                targets_wdl = torch.sigmoid(batch_targets / 4.0)
                loss_wdl = (loss_wdl_fn(preds_wdl, targets_wdl) * weights).mean()
                
                loss_pawns = (loss_pawns_fn(score_preds, batch_targets) * weights).mean()
                
                targets_mate = (torch.abs(batch_targets) > 15.0).float()
                loss_mate = (loss_mate_fn(mate_preds, targets_mate) * weights).mean()
                
                loss = (0.8 * loss_wdl) + (0.2 * loss_pawns) + (alpha * loss_mate)

            # Aktualizacja wag (Backpropagation)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            # Przycięcie gradientów (Grad Clip) zapobiega eksplozji i psuciu się modelu
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            scaler.step(optimizer)
            scaler.update()
            
            scheduler.step()

            running_train_loss += loss.item()
            steps_since_val += 1

            current_lr = optimizer.param_groups[0]['lr']
            train_pbar.set_postfix({'loss': f"{loss.item():.4f}", 'lr': f"{current_lr:.6f}"})

            # WALIDACJA ŚRÓDEPOKOWA (Zapisujemy częściej niż raz na epokę)
            if (step + 1) % val_interval == 0 or (step + 1) == total_batches:
                avg_train_loss = running_train_loss / steps_since_val
                
                model.eval()
                val_loss = 0.0
                total_val_mae = 0.0      
                total_val_pos = 0        
                
                val_pbar = tqdm(val_loader, desc=f"Epoka {epoch+1} [Val] Krok {step+1}", leave=False)
                val_steps_taken = 0
                
                with torch.no_grad():
                    for val_data in val_pbar:
                        if val_steps_taken >= MAX_VALIDATION_BATCHES: break 
                        boards, turns, castling, eps, val_targets = [x.to(device, non_blocking=True) for x in val_data]
                        val_inputs = build_batch_on_gpu(boards, turns, castling, eps)
                        
                        val_targets = torch.where(turns == 0, -val_targets, val_targets)

                        v_score_preds, v_mate_preds = model(val_inputs)
                        
                        v_weights = torch.ones_like(val_targets)
                        v_weights[torch.abs(val_targets) >= 8.0] = 4.0
                        
                        v_preds_wdl = torch.sigmoid(v_score_preds / 4.0)
                        v_targets_wdl = torch.sigmoid(val_targets / 4.0)
                        v_loss_wdl = (loss_wdl_fn(v_preds_wdl, v_targets_wdl) * v_weights).mean()
                        
                        v_loss_pawns = (loss_pawns_fn(v_score_preds, val_targets) * v_weights).mean()
                        
                        v_targets_mate = (torch.abs(val_targets) > 15.0).float()
                        v_loss_mate = (loss_mate_fn(v_mate_preds, v_targets_mate) * v_weights).mean()
                        
                        v_loss = (0.8 * v_loss_wdl) + (0.2 * v_loss_pawns) + (alpha * v_loss_mate)
                        val_loss += v_loss.item()
                        
                        total_val_mae += torch.abs(v_score_preds - val_targets).sum().item()
                        total_val_pos += val_targets.size(0)
                        
                        val_steps_taken += 1
                        val_pbar.set_postfix({'loss': f"{v_loss.item():.4f}"})

                avg_val_loss = val_loss / val_steps_taken
                avg_val_mae = total_val_mae / total_val_pos 

                print(f"\n\t[Epoka {epoch+1} | Krok {step+1}/{total_batches}] Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val MAE: {avg_val_mae:.2f}")

                # ZAPISANIE NOWYCH WAG, TYLKO JEŚLI JEST POPRAWA
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    epochs_no_improve = 0
                    
                    # 1. Zapis samych wag dla ONNX
                    torch.save(model.state_dict(), "best_chess_model.pth")
                    
                    # 2. Pełny zrzut pamięci do wznowienia w razie awarii prądu
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'best_val_loss': best_val_loss
                    }, "checkpoint.pth")
                    
                    print(f"\t>>> Zapisano nowy najlepszy model! (Loss: {best_val_loss:.4f} | MAE: {avg_val_mae:.2f})")
                else:
                    epochs_no_improve += 1

                # Wczesne zatrzymanie: jeśli przez pewien czas nie ma poprawy, przerywamy pętlę
                if epochs_no_improve >= patience:
                    print(f"\n🛑 WCZESNE ZATRZYMANIE (EARLY STOPPING)!")
                    model.load_state_dict(torch.load("best_chess_model.pth"))
                    model.eval()
                    return model

                running_train_loss = 0.0
                steps_since_val = 0
                model.train()

    model.load_state_dict(torch.load("best_chess_model.pth"))
    model.eval()
    print("\nPomyślnie wczytano najlepszą wersję modelu.")
    
    return model
