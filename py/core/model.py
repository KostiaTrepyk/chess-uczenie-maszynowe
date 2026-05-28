import torch
import math
import torch.nn as nn
from torch.amp import GradScaler
from torch.utils.data import DataLoader
from typing import List

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
            nn.Conv2d(15, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1)
        )

        # Динамически создаем нужное количество блоков с помощью генератора
        self.resnet_blocks = nn.Sequential(
            *[ResidualBlock(128) for _ in range(num_blocks)]
        )

        self.value_head = nn.Sequential(
            nn.Conv2d(128, 8, kernel_size=1),
            nn.BatchNorm2d(8),
            nn.LeakyReLU(0.1),
            nn.Flatten(),
            nn.Linear(8 * 8 * 8, 256),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Tanh()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_conv(x)
        x = self.resnet_blocks(x)
        return self.value_head(x)

PIECE_TO_CHANNEL = {
    'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5,
    'p': 6, 'n': 7, 'b': 8, 'r': 9, 'q': 10, 'k': 11
}

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
    loss_fn = nn.HuberLoss()
    epochs = 100

    patience = 5
    best_val_loss = float('inf')
    epochs_no_improve = 0

    scaler = GradScaler(device)
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
                loss = loss_fn(predictions, batch_targets)

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

        print(f"Эпоха {epoch+1} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

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

    # Создаем фейковый тензор нужной формы (1 батч, 14 слоев, 8x8)
    dummy_input = torch.randn(1, 14, 8, 8, device=device)

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

    with torch.no_grad():
        predictions = model(inputs).view(-1).tolist()

    # Разворачиваем tanh обратно в пешки
    eval_in_pawns = []
    for p in predictions:
        # Защита от бесконечности (если сеть выдаст ровно 1.0 или -1.0)
        p_clipped = max(-0.999, min(0.999, p))
        eval_in_pawns.append(math.atanh(p_clipped) * 4.0)

    return eval_in_pawns

def evaluate_model_metrics(model: torch.nn.Module, dataloader: torch.utils.data.DataLoader, device: torch.device):
    """
    Прогоняет датасет через модель и возвращает среднюю ошибку в пешках и точность определения лидера.
    """
    model.eval()
    total_mae_pawns = 0.0
    correct_signs = 0
    total_positions = 0

    with torch.no_grad():
        for batch_inputs, batch_targets in dataloader:
            batch_inputs = batch_inputs.to(device, non_blocking=True)
            batch_targets = batch_targets.to(device, non_blocking=True)

            # Получаем предсказания сети в сжатом виде [-1, 1]
            preds = model(batch_inputs).view(-1)
            targets = batch_targets.view(-1)

            # Защита от бесконечности перед atanh
            preds_clipped = torch.clamp(preds, min=-0.999, max=0.999)
            targets_clipped = torch.clamp(targets, min=-0.999, max=0.999)

            # Денормализация: возвращаем значения обратно в пешки (Разворачиваем tanh)
            preds_pawns = torch.atanh(preds_clipped) * 4.0
            targets_pawns = torch.atanh(targets_clipped) * 4.0

            # 1. Считаем ошибку в пешках
            mae = torch.abs(preds_pawns - targets_pawns).sum().item()
            total_mae_pawns += mae

            # 2. Считаем совпадение знаков (Угадали ли, кто побеждает)
            # Если оба числа > 0 (белые) или оба < 0 (черные) или оба == 0
            preds_signs = torch.sign(preds_pawns)
            targets_signs = torch.sign(targets_pawns)
            correct_signs += (preds_signs == targets_signs).sum().item()

            total_positions += targets.size(0)

    # Итоговые метрики
    avg_mae_pawns = total_mae_pawns / total_positions
    sign_accuracy = (correct_signs / total_positions) * 100.0

    print(f"\n📊 Результаты тестирования ({total_positions} позиций):")
    print(f"Оценка лидера (Sign Accuracy): {sign_accuracy:.1f}%")
    print(f"Средняя ошибка (MAE):          {avg_mae_pawns:.2f} пешек")

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
