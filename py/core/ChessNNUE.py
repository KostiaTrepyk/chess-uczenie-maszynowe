import os
import csv
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from tqdm import tqdm

checks_per_epoch = 1
max_val_batches = 300

CHAR_TO_W = {'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'p': 5, 'n': 6, 'b': 7, 'r': 8, 'q': 9}
CHAR_TO_B = {'p': 0, 'n': 1, 'b': 2, 'r': 3, 'q': 4, 'P': 5, 'N': 6, 'B': 7, 'R': 8, 'Q': 9}

def parse_fen_for_nnue(fen: str):
    """
    Новый парсер: Возвращает индексы фигур + Вектор глобального состояния (Рокировки, Очередь, EP)
    """
    parts = fen.split()
    board_part = parts[0]
    
    white_king_sq = None
    black_king_sq = None
    pieces = []
    
    rank, file = 7, 0 
    for char in board_part:
        if char == '/':
            rank -= 1
            file = 0
        elif char.isdigit():
            file += int(char)
        else:
            sq = rank * 8 + file
            if char == 'K': white_king_sq = sq
            elif char == 'k': black_king_sq = sq
            else: pieces.append((char, sq))
            file += 1
            
    if white_king_sq is None or black_king_sq is None:
        return [], [], []
        
    black_king_sq_flipped = black_king_sq ^ 56
    w_indices, b_indices = [], []
    
    for char, sq in pieces:
        w_indices.append((white_king_sq * 640) + (CHAR_TO_W[char] * 64) + sq)
        b_indices.append((black_king_sq_flipped * 640) + (CHAR_TO_B[char] * 64) + (sq ^ 56))
        
    # Формируем глобальное состояние (6 дополнительных фичей)
    turn = 1.0 if len(parts) > 1 and parts[1] == 'w' else 0.0
    castling = parts[2] if len(parts) > 2 else '-'
    ep = 1.0 if len(parts) > 3 and parts[3] != '-' else 0.0
    
    state = [
        turn,
        1.0 if 'K' in castling else 0.0,
        1.0 if 'Q' in castling else 0.0,
        1.0 if 'k' in castling else 0.0,
        1.0 if 'q' in castling else 0.0,
        ep
    ]
        
    return w_indices, b_indices, state

class ChessNNUE(nn.Module):
    def __init__(self):
        super().__init__()
        
        # 1. Возвращаем полноценную память (256 каналов)
        self.feature_transformer = nn.EmbeddingBag(
            num_embeddings=41025, 
            embedding_dim=256, 
            mode='sum', 
            padding_idx=41024
        )
        
        # 2. Утяжеленная глубокая архитектура (518 -> 512 -> 128 -> 32 -> 1)
        self.fc1 = nn.Linear(518, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 32)
        self.output = nn.Linear(32, 1)

        # Инициализация Kaiming отлично подходит для ReLU
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, w_indices, b_indices, state):
        # Аккумулятор оставляем с обычным Clipped ReLU для стабильности
        acc_w = torch.clamp(self.feature_transformer(w_indices), 0.0, 1.0)
        acc_b = torch.clamp(self.feature_transformer(b_indices), 0.0, 1.0)
        
        x = torch.cat([acc_w, acc_b, state], dim=1)
        
        # 3. Применяем Squared Clipped ReLU для всех скрытых слоев
        x = torch.pow(torch.clamp(self.fc1(x), 0.0, 1.0), 2)
        x = torch.pow(torch.clamp(self.fc2(x), 0.0, 1.0), 2)
        x = torch.pow(torch.clamp(self.fc3(x), 0.0, 1.0), 2)

        return self.output(x)

class NNUEChessDataset(Dataset):
    def __init__(self, data_file_path: str):
        self._load_data(data_file_path)

    def _load_data(self, path):
        # Используем новый файл кэша, чтобы не смешивать со старым форматом
        cache_path = path + ".cache_v2.pt" 
        
        if os.path.exists(cache_path):
            print(f"🚀 Znaleziono plik cache v2: {cache_path}.")
            cache_dict = torch.load(cache_path, weights_only=False)
            self.w_indices = cache_dict['w']
            self.b_indices = cache_dict['b']
            self.states = cache_dict['states']
            self.targets = cache_dict['targets']
            self.num_samples = len(self.targets)
            return

        with open(path, 'r', encoding='utf-8') as f:
            num_lines = sum(1 for _ in f)

        self.w_indices = torch.full((num_lines, 32), 41024, dtype=torch.int32)
        self.b_indices = torch.full((num_lines, 32), 41024, dtype=torch.int32)
        self.states = torch.zeros((num_lines, 6), dtype=torch.float32)
        self.targets = torch.zeros((num_lines, 1), dtype=torch.float32)

        valid_idx = 0
        with open(path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            for row in tqdm(reader, total=num_lines, desc="Przetwarzanie FEN"):
                if len(row) < 2: continue
                try:
                    score_str = row[1].strip()
                    cp_score = 2000 if '#' in score_str and not score_str.startswith('-') else -2000 if '#' in score_str else int(float(score_str))
                    cp_score = max(-2000, min(2000, cp_score))
                    pawn_score = cp_score / 100.0
                    
                    w_idx, b_idx, state = parse_fen_for_nnue(row[0].strip())
                    if not w_idx or not b_idx: continue
                    
                    length = min(len(w_idx), 32)
                    self.w_indices[valid_idx, :length] = torch.IntTensor(w_idx[:length])
                    self.b_indices[valid_idx, :length] = torch.IntTensor(b_idx[:length])
                    self.states[valid_idx] = torch.FloatTensor(state)
                    self.targets[valid_idx, 0] = pawn_score 
                    
                    valid_idx += 1
                except ValueError:
                    continue
                    
        self.w_indices = self.w_indices[:valid_idx]
        self.b_indices = self.b_indices[:valid_idx]
        self.states = self.states[:valid_idx]
        self.targets = self.targets[:valid_idx]
        self.num_samples = valid_idx

        torch.save({'w': self.w_indices, 'b': self.b_indices, 'states': self.states, 'targets': self.targets}, cache_path) 

    def __len__(self): return self.num_samples

    def __getitem__(self, idx):
        return (
            self.w_indices[idx].to(torch.long),
            self.b_indices[idx].to(torch.long),
            self.states[idx],
            self.targets[idx],
        )

def train_nnue(train_loader, val_loader) -> ChessNNUE:
    device_str = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device_str)
    print(f"\nTrenowanie NNUE na urządzeniu: {device}")

    torch.backends.cudnn.benchmark = True
    model = ChessNNUE().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)
    loss_fn = nn.SmoothL1Loss() 
    epochs = 150
    patience = 30 * checks_per_epoch  
    best_val_loss = float('inf')
    epochs_no_improve = 0
    total_batches = len(train_loader)
    val_interval = max(1, total_batches // checks_per_epoch)
    scaler = torch.amp.GradScaler('cuda')
    
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=0.01, epochs=epochs,
        steps_per_epoch=total_batches, pct_start=0.1,
        div_factor=10.0, final_div_factor=1000.0
    )

    for epoch in range(epochs):
        model.train()
        running_train_loss = 0.0
        steps_since_val = 0
        train_pbar = tqdm(train_loader, desc=f"Epoka {epoch+1}/{epochs} [Train]", leave=False)
        
        for step, batch_data in enumerate(train_pbar):
            w_idx, b_idx, states, targets = [x.to(device, non_blocking=True) for x in batch_data]
            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast('cuda'):
                predictions = model(w_idx, b_idx, states)
                loss = loss_fn(predictions, targets)

            scaler.scale(loss).backward()
            if scaler.is_enabled():
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            
            running_train_loss += loss.item()
            steps_since_val += 1
            current_lr = scheduler.get_last_lr()[0]
            train_pbar.set_postfix({'loss': f"{loss.item():.4f}", 'lr': f"{current_lr:.6f}"})

            if (step + 1) % val_interval == 0 or (step + 1) == total_batches:
                avg_train_loss = running_train_loss / steps_since_val
                model.eval()
                val_loss = 0.0
                val_pbar = tqdm(val_loader, desc=f"Epoka {epoch+1} [Val] Krok {step+1}", leave=False)
                val_steps_taken = 0
                
                with torch.no_grad():
                    for val_data in val_pbar:
                        if val_steps_taken >= max_val_batches: break 
                        v_w_idx, v_b_idx, v_states, v_targets = [x.to(device, non_blocking=True) for x in val_data]
                        val_preds = model(v_w_idx, v_b_idx, v_states)
                        v_loss = loss_fn(val_preds, v_targets)
                        val_loss += v_loss.item()
                        val_steps_taken += 1
                        val_pbar.set_postfix({'loss': f"{v_loss.item():.4f}"})

                avg_val_loss = val_loss / val_steps_taken
                print(f"\n\t[Epoka {epoch+1} | Krok {step+1}/{total_batches}] Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    epochs_no_improve = 0
                    torch.save(model.state_dict(), "best_nnue_model.pth")
                    print(f"\t>>> Zapisano nowy najlepszy model! (Loss: {best_val_loss:.4f})")
                else:
                    epochs_no_improve += 1

                if epochs_no_improve >= patience:
                    print(f"\n🛑 WCZESNE ZATRZYMANIE (EARLY STOPPING)!")
                    model.load_state_dict(torch.load("best_nnue_model.pth"))
                    model.eval()
                    return model

                running_train_loss = 0.0
                steps_since_val = 0
                model.train() 

    model.load_state_dict(torch.load("best_nnue_model.pth"))
    model.eval()
    return model


def evaluate_nnue(model: ChessNNUE, dataloader, device: torch.device) -> tuple[float, float]:
    model.eval()
    correct_signs = 0
    total_positions = 0
    total_mae_pawns = 0.0

    # Настройки для диапазонов (bins) в пешках
    bins = [0, 2, 5, 7, 10, 15, 100]
    bin_diffs_sum = {i: 0.0 for i in range(len(bins)-1)}
    bin_counts = {i: 0 for i in range(len(bins)-1)}
    labels = ["[-2..2]", "(2..5]", "(5..7]", "(7..10]", "(10..15]", "(15..100]"]

    eval_pbar = tqdm(dataloader, desc="Testowanie NNUE", leave=True)

    print("\n--- Сравнение: Предсказание vs Реальность (w pionach) ---")
    print(f"{'Predykcja':<18} | {'Prawdziwa ocena':<18}")
    print("-" * 40)

    with torch.no_grad():
        for i, batch_data in enumerate(eval_pbar):
            w_idx, b_idx, states, targets_pawns = [x.to(device, non_blocking=True) for x in batch_data]

            preds_pawns = model(w_idx, b_idx, states)

            # Вывод первых позиций для визуального контроля (в пешках)
            if i == 0:
                for j in range(min(15, targets_pawns.size(0))):
                    print(f"{preds_pawns[j].item():>18.2f} | {targets_pawns[j].item():>18.2f}")
                print("-" * 40)

            diffs = torch.abs(preds_pawns - targets_pawns).squeeze()
            abs_targets = torch.abs(targets_pawns).squeeze()

            # Защита от батча размером 1
            if diffs.dim() == 0:
                diffs = diffs.unsqueeze(0)
                abs_targets = abs_targets.unsqueeze(0)

            # Векторное распределение по корзинам (очень быстро на GPU)
            for j in range(len(bins)-1):
                low, high = bins[j], bins[j+1]
                if j == 0:
                    mask = (abs_targets <= high)
                else:
                    mask = (abs_targets > low) & (abs_targets <= high)
                
                bin_diffs_sum[j] += diffs[mask].sum().item()
                bin_counts[j] += mask.sum().item()

            # Общая статистика
            mae = diffs.sum().item()
            total_mae_pawns += mae

            preds_signs = torch.sign(preds_pawns)
            targets_signs = torch.sign(targets_pawns)
            correct_signs += (preds_signs == targets_signs).sum().item()

            total_positions += targets_pawns.size(0)

            running_mae = total_mae_pawns / total_positions
            running_acc = (correct_signs / total_positions) * 100.0
            eval_pbar.set_postfix({'MAE': f"{running_mae:.3f}", 'Acc': f"{running_acc:.1f}%"})

    avg_mae_pawns = total_mae_pawns / total_positions
    sign_accuracy = (correct_signs / total_positions) * 100.0

    # === ВЫВОД РАЗНИЦЫ ПО ДИАПАЗОНАМ ===
    print("\n" + "="*75)
    print(f"📊 Statystyki szczegółowe (na podstawie {total_positions} pozycji):")

    for j, label in enumerate(labels):
        count = bin_counts[j]
        if count > 0:
            avg_diff = bin_diffs_sum[j] / count
            print(f"  {label:10} Średni błąd: {avg_diff:>5.2f} piona  (Pozycji: {count})")
        else:
            print(f"  {label:10} Brak pozycji (Pozycji: 0)")

    print("="*75 + "\n")

    print(f"📈 Podsumowanie całkowite:")
    print(f"\tOcena przewagi (Sign Accuracy): {sign_accuracy:.1f}%")
    print(f"\tŚredni błąd (MAE):          {avg_mae_pawns:.3f} piona")

    return avg_mae_pawns, sign_accuracy
