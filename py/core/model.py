import torch
import math
import torch.nn as nn
from torch.amp import GradScaler
from torch.utils.data import DataLoader
from typing import List

class TransformerBlock(nn.Module):
    def __init__(self, channels: int, heads: int = 4):
        super().__init__()
        # Внимание: 4 "головы" будут искать разные паттерны (связки, защиту короля и т.д.)
        self.attention = nn.MultiheadAttention(embed_dim=channels, num_heads=heads, batch_first=True)
        self.norm = nn.LayerNorm(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.size()
        
        # Разворачиваем доску 8x8 в последовательность из 64 клеток для трансформера
        x_flat = x.view(b, c, h * w).permute(0, 2, 1) # Форма: (Batch, 64, Channels)
        
        # Сеть "смотрит" сама на себя, находя скрытые связи между клетками
        attn_out, _ = self.attention(x_flat, x_flat, x_flat)
        
        # Добавляем исходные данные и нормализуем (Residual connection)
        out = self.norm(x_flat + attn_out)
        
        # Сворачиваем обратно в классическую форму доски
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

class FocusMSELoss(nn.Module):
    def __init__(self, focus_strength=4.0):
        super().__init__()
        self.focus_strength = focus_strength
        self.mse = nn.MSELoss(reduction='none') # Считаем ошибку для каждого элемента отдельно

    def forward(self, predictions, targets):
        # Базовая ошибка
        base_loss = self.mse(predictions, targets)
        
        # Усилитель: Максимален при targets == 0.5 (равная игра), минимален по краям
        # При focus_strength=4.0, ошибки в равных позициях штрафуются в 2 раза сильнее
        weight = 1.0 + self.focus_strength * targets * (1.0 - targets)
        
        # Умножаем ошибку на вес и возвращаем среднее
        return torch.mean(base_loss * weight)

# *** Конвертация FEN в тензор и обучение ***

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

    # Обучаем
    model = ChessResNet(num_blocks=10).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
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

        for batch_inputs, batch_targets in train_loader:
            batch_inputs = batch_inputs.to(device, non_blocking=True)
            batch_targets = batch_targets.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True) # set_to_none=True работает быстрее обычного zero_grad()

            # 2. Оборачиваем forward pass и расчет loss в autocast
            with torch.amp.autocast(device_str):
                predictions = model(batch_inputs)
                loss = loss_fn(predictions, batch_targets.long())

            # 3. Масштабируем градиенты для защиты от "исчезновения" 16-битных чисел
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)

        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for val_inputs, val_targets in val_loader:
                # ПЕРЕНОСИМ ВАЛИДАЦИОННЫЕ БАТЧИ НА GPU
                val_inputs = val_inputs.to(device, non_blocking=True)
                val_targets = val_targets.to(device, non_blocking=True)

                val_preds = model(val_inputs)
                v_loss = loss_fn(val_preds, val_targets)
                val_loss += v_loss.item()

        avg_val_loss = val_loss / len(val_loader)

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

    # Создаем шкалу корзин от -10 до +10 пешек (100 классов)
    bucket_values = torch.linspace(-10.0, 10.0, steps=100, device=device)

    with torch.no_grad():
        for batch_inputs, batch_targets in dataloader:
            batch_inputs = batch_inputs.to(device, non_blocking=True)
            batch_targets = batch_targets.to(device, non_blocking=True) # Форма: [2048]

            # 1. Получаем логиты от сети
            logits = model(batch_inputs) # Форма: [2048, 100]

            # 2. Переводим логиты в вероятности (Softmax)
            probabilities = torch.softmax(logits, dim=-1)

            # 3. Предсказания сети: Считаем матожидание (Вероятность * Значение корзины)
            preds_pawns = torch.sum(probabilities * bucket_values, dim=-1) # Форма: [2048]

            # 4. Реальная оценка: Вытаскиваем значение пешек по индексу правильного класса
            targets_pawns = bucket_values[batch_targets] # Форма: [2048]

            # 5. Считаем ошибку (MAE)
            mae = torch.abs(preds_pawns - targets_pawns).sum().item()
            total_mae_pawns += mae

            # 6. Точность угадывания лидера (сравниваем знаки)
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

# Функция конвертации WDL -> Пешки для тензоров (работает на GPU)
def tensor_wdl_to_pawns(wdl_tensor: torch.Tensor) -> torch.Tensor:
    # Ограничиваем, чтобы не получить log10(0) или деление на ноль
    safe_wdl = torch.clamp(wdl_tensor, min=0.001, max=0.999)
    # Обратная формула WDL: eval = -4 * log10(1/WDL - 1)
    return -4.0 * torch.log10((1.0 / safe_wdl) - 1.0)