import torch
import torch.nn as nn
from torch.amp import GradScaler
from torch.utils.data import DataLoader
from typing import List
from tqdm import tqdm
import torch.nn.functional as F

class TransformerBlock(nn.Module):
    def __init__(self, channels: int, heads: int = 4):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim=channels, num_heads=heads, batch_first=True)
        self.norm = nn.LayerNorm(channels)
        # Обучаемые позиционные эмбеддинги для 64 клеток доски
        self.pos_embedding = nn.Parameter(torch.randn(1, 64, channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.size()
        
        x_flat = x.view(b, c, h * w).permute(0, 2, 1) # (Batch, 64, Channels)
        
        # Добавляем позиционную информацию
        x_flat = x_flat + self.pos_embedding
        
        attn_out, _ = self.attention(x_flat, x_flat, x_flat)
        out = self.norm(x_flat + attn_out)
        
        return out.permute(0, 2, 1).view(b, c, h, w)
    
class ResidualBlock(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        self.leaky_relu = nn.LeakyReLU(0.1)

        # Squeeze-and-Excitation блок
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), # Сжимаем 8x8 до 1x1 (Глобальный контекст)
            nn.Conv2d(channels, channels // reduction, kernel_size=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(channels // reduction, channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x

        out = self.leaky_relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        # Применяем механизм внимания
        out = out * self.se(out)

        out += residual
        return self.leaky_relu(out)

class ChessResNet(nn.Module):
    def __init__(self, num_blocks=10):
        super().__init__()
        self.input_conv = nn.Sequential(
            nn.Conv2d(15, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1)
        )

        self.resnet_blocks = nn.Sequential(
            *[ResidualBlock(256) for _ in range(num_blocks)]
        )

        # <-- ДОБАВЛЯЕМ ТРАНСФОРМЕР СЮДА -->
        self.transformer = TransformerBlock(channels=256, heads=4)

        self.value_head = nn.Sequential(
            nn.Conv2d(256, 16, kernel_size=1), 
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.1),
            nn.Flatten(),
            nn.Linear(16 * 8 * 8, 512), 
            nn.LeakyReLU(0.1),
            nn.Dropout(0.3),
            nn.Linear(512, 100),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_conv(x)
        x = self.resnet_blocks(x)
        x = self.transformer(x) # <-- ПРОПУСКАЕМ ЧЕРЕЗ ТРАНСФОРМЕР
        return self.value_head(x)

PIECE_TO_CHANNEL = {
    'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5,
    'p': 6, 'n': 7, 'b': 8, 'r': 9, 'q': 10, 'k': 11
}

# *** Конвертация FEN в тензор и обучение ***

def build_batch_on_gpu(boards, turns, castling, ep):
    """Мгновенно собирает батч [Batch, 15, 8, 8] прямо на видеокарте"""
    b = boards.size(0)
    device = boards.device
    
    # 1. Фигуры (12 слоев) через one_hot 
    pieces = F.one_hot(boards, num_classes=13)[:, :, :12].float()
    pieces = pieces.permute(0, 2, 1).view(b, 12, 8, 8)
    
    # 2. Очередь хода (1 слой) размножаем на всю доску
    turns_layer = turns.view(b, 1, 1, 1).expand(b, 1, 8, 8)
    
    # 3. Рокировка (1 слой)
    castling_layer = torch.zeros((b, 1, 8, 8), device=device, dtype=torch.float32)
    castling_layer[:, 0, 7, 7] = castling[:, 0]
    castling_layer[:, 0, 7, 0] = castling[:, 1]
    castling_layer[:, 0, 0, 7] = castling[:, 2]
    castling_layer[:, 0, 0, 0] = castling[:, 3]
    
    # 4. Взятие на проходе (1 слой)
    ep_layer = torch.zeros((b, 1, 64), device=device, dtype=torch.float32)
    has_ep = ep > 0
    ep_idx = ep[has_ep] - 1
    ep_layer[has_ep, 0, ep_idx] = 1.0 # Вставляем единицы только там, где есть En Passant
    ep_layer = ep_layer.view(b, 1, 8, 8)
    
    # Склеиваем 12 + 1 + 1 + 1 = 15 слоев
    return torch.cat([pieces, turns_layer, castling_layer, ep_layer], dim=1)

def fen_to_tensor(fen: str) -> torch.Tensor:
    parts = fen.split()
    board_part = parts[0]
    turn_part = parts[1] if len(parts) > 1 else 'w'
    castling_part = parts[2] if len(parts) > 2 else '-'
    ep_part = parts[3] if len(parts) > 3 else '-' # <-- Читаем En Passant

    # Инициализируем тензор (ТЕПЕРЬ 15 СЛОЕВ)
    tensor = torch.zeros((15, 8, 8), dtype=torch.float32)

    # 1. Слой 0-11: Фигуры
    row, col = 0, 0
    for char in board_part:
        if char == '/':
            row += 1
            col = 0
        elif char.isdigit():
            col += int(char)
        else:
            channel = PIECE_TO_CHANNEL[char]
            tensor[channel, row, col] = 1.0
            col += 1

    # 2. Слой 12: Очередь хода
    if turn_part == 'w':
        tensor[12, :, :] = 1.0

    # 3. Слой 13: Рокировка
    if 'K' in castling_part: tensor[13, 7, 7] = 1.0
    if 'Q' in castling_part: tensor[13, 7, 0] = 1.0
    if 'k' in castling_part: tensor[13, 0, 7] = 1.0
    if 'q' in castling_part: tensor[13, 0, 0] = 1.0

    # 4. Слой 14: Взятие на проходе
    if ep_part != '-':
        # Переводим букву в колонку (a=0, e=4, h=7)
        ep_col = ord(ep_part[0]) - ord('a')
        # Переводим цифру в строку (8=0, 3=5, 1=7)
        ep_row = 8 - int(ep_part[1])
        tensor[14, ep_row, ep_col] = 1.0

    return tensor

def train_model(train_loader : DataLoader, val_loader : DataLoader) -> ChessResNet:
    device_str = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nОбучение на устройстве: {device}")

    # Включаем аппаратное ускорение для сверточных сетей (Дает +10-15% скорости)
    torch.backends.cudnn.benchmark = True

    model = ChessResNet(num_blocks=10).to(device)
    
    # Компиляция модели для PyTorch 2.0+ (Дает еще +20% скорости). Если выдаст ошибку на Windows - просто удали эту строку.
    # model = torch.compile(model) 

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()
    epochs = 100

    patience = 5
    best_val_loss = float('inf')
    epochs_no_improve = 0

    scaler = GradScaler()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2
    )

    print("\nНачинаем обучение...")
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        train_pbar = tqdm(train_loader, desc=f"Эпоха {epoch+1}/{epochs} [Train]", leave=False)
        for batch_data in train_pbar:
            # 1. Переносим сырые легкие данные на видеокарту
            boards, turns, castling, eps, batch_targets = [x.to(device, non_blocking=True) for x in batch_data]
            
            # 2. Мгновенно собираем тяжелые матрицы на GPU
            batch_inputs = build_batch_on_gpu(boards, turns, castling, eps)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(device_str):
                # PyTorch автоматически применит FlashAttention внутри Трансформера
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

        val_pbar = tqdm(val_loader, desc=f"Эпоха {epoch+1}/{epochs} [Val]", leave=False)
        with torch.no_grad():
            for val_data in val_pbar:
                # То же самое для валидации: переносим и собираем
                boards, turns, castling, eps, val_targets = [x.to(device, non_blocking=True) for x in val_data]
                val_inputs = build_batch_on_gpu(boards, turns, castling, eps)

                val_preds = model(val_inputs)
                v_loss = loss_fn(val_preds, val_targets)
                val_loss += v_loss.item()
                
                val_pbar.set_postfix({'loss': f"{v_loss.item():.4f}"})

        avg_val_loss = val_loss / len(val_loader)

        # Оставляем только финальный вывод, полоски исчезнут (leave=False)
        print(f"\tЭпоха {epoch+1} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        scheduler.step(avg_val_loss)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), "best_chess_model.pth")
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print(f"\n🛑 РАННЯЯ ОСТАНОВКА!")
            break

    model.load_state_dict(torch.load("best_chess_model.pth"))
    model.eval()
    print("Успешно загружена лучшая версия модели.")
    
    return model

# *** Загрузка модели и экспорт ***

def load_model()-> ChessResNet:
    # Загрузить готовую модель
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 2. Создаем "пустую" архитектуру сети и переносим на устройство
    model = ChessResNet(num_blocks=10).to(device)

    # 3. Загружаем веса из файла
    # map_location гарантирует, что модель загрузится даже если обучалась на GPU, а запускается на CPU
    model.load_state_dict(torch.load("best_chess_model.pth", map_location=device))

    # 4. Обязательно переводим в режим предсказания
    model.eval()

    return model

def export_model_to_onnx():
    device = torch.device('cpu')
    model = ChessResNet(num_blocks=10).to(device)
    model.load_state_dict(torch.load("best_chess_model.pth", map_location=device))
    model.eval()

    # Создаем фейковый тензор нужной формы (1 батч, 15 слоев, 8x8)
    dummy_input = torch.randn(1, 15, 8, 8, device=device)

    # Экспортируем
    torch.onnx.export(
        model,
        dummy_input,
        "chess_model.onnx",
        export_params=True,
        opset_version=18,
        input_names=['input'],
        output_names=['output']
    )
    print("✅ Файл chess_model.onnx готов!")

# *** Предсказание оценки ***

def predict_evaluation(model: nn.Module, fens: List[str]) -> List[float]:
    if not fens:
        return []

    model.eval()
    device = next(model.parameters()).device
    inputs = torch.stack([fen_to_tensor(fen) for fen in fens]).to(device)

    # Массив значений для каждой корзины от -10 до +10
    bucket_values = torch.linspace(-10.0, 10.0, steps=100, device=device)

    with torch.no_grad():
        # Получаем сырые логиты
        logits = model(inputs)
        # Переводим логиты в реальные проценты вероятности (сумма = 1.0)
        probabilities = torch.softmax(logits, dim=-1)
        
        # Математическое ожидание: Вероятность * Значение корзины
        # Это даст идеальную дробную точность (например, 1.27 пешек)
        expected_evals = torch.sum(probabilities * bucket_values, dim=-1).tolist()

    return expected_evals

def evaluate_model_metrics(model: torch.nn.Module, dataloader: torch.utils.data.DataLoader, device: torch.device) -> tuple[float, float]:
    model.eval()
    total_mae_pawns = 0.0
    correct_signs = 0
    total_positions = 0
    bucket_values = torch.linspace(-10.0, 10.0, steps=100, device=device)

    with torch.no_grad():
        # === ИЗМЕНЕНИЯ ЗДЕСЬ ===
        for val_data in dataloader:
            # 1. Распаковываем 5 элементов
            boards, turns, castling, eps, batch_targets = [x.to(device, non_blocking=True) for x in val_data]
            
            # 2. Собираем батч на GPU
            batch_inputs = build_batch_on_gpu(boards, turns, castling, eps)

            # 3. Передаем в модель
            logits = model(batch_inputs) 
        # =======================

            probabilities = torch.softmax(logits, dim=-1)
            preds_pawns = torch.sum(probabilities * bucket_values, dim=-1) 
            targets_pawns = bucket_values[batch_targets] 

            mae = torch.abs(preds_pawns - targets_pawns).sum().item()
            total_mae_pawns += mae

            preds_signs = torch.sign(preds_pawns)
            targets_signs = torch.sign(targets_pawns)
            correct_signs += (preds_signs == targets_signs).sum().item()

            total_positions += batch_targets.size(0)

    avg_mae_pawns = total_mae_pawns / total_positions
    sign_accuracy = (correct_signs / total_positions) * 100.0

    print(f"\n📊 Результаты тестирования ({total_positions} позиций):")
    print(f"\tОценка лидера (Sign Accuracy): {sign_accuracy:.1f}%")
    print(f"\tСредняя ошибка (MAE):          {avg_mae_pawns:.2f} пешек")

    return avg_mae_pawns, sign_accuracy

def show_model_stats(model, val_loader, df):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    test_positions = [
        [df["FEN"].iloc[i], df["Evaluation"].iloc[i]] for i in range(10)
    ]
    
    # 1. Достаем только FEN-строки для нейросети
    just_fens = [item[0] for item in test_positions]

    # 2. Делаем предикт
    evaluations = predict_evaluation(model, just_fens)

    # 3. Выводим результат
    print("\nРезультаты оценки нейросетью:")
    for [fen, correct_eval], eval_score in zip(test_positions, evaluations):
        print(f"\nПозиция: {fen.split()[0]}")
        print(f"Оценка сети: {eval_score:.2f} пешек (или {int(eval_score * 100)} сантипешек) | Реальная: {correct_eval}")

    # СТАТИСТИКА
    evaluate_model_metrics(model, val_loader, device)
