import torch
import torch.nn as nn
from tqdm import tqdm
from typing import List

from core.features import fen_to_tensor, build_batch_on_gpu
from core.consts import WDL_SCALE

def prob_to_pawns(prob_tensor: torch.Tensor) -> torch.Tensor:
    """Обратная конвертация: Вероятность (0..1) -> Пешки с защитой от выбросов."""
    # Защита от деления на ноль / логарифма нуля (обрезаем 0.001 ... 0.999)
    p = torch.clamp(prob_tensor, min=1e-4, max=1.0 - 1e-4)
    pawns = -WDL_SCALE * torch.log((1.0 / p) - 1.0)
    
    # ЖЕСТКАЯ ОБРЕЗКА: Не даем пешкам улетать в бесконечность при 99% вероятности
    return torch.clamp(pawns, min=-10.0, max=10.0)

def predict_evaluation(model: nn.Module, fens: List[str]) -> List[float]:
    if not fens:
        return []

    model.eval()
    device = next(model.parameters()).device
    inputs = torch.stack([fen_to_tensor(fen) for fen in fens]).to(device)

    with torch.no_grad():
        out = model(inputs)
        # collapse extra output dims to a single scalar per sample (robust to different value_head shapes)
        if out.ndim > 1:
            out = out.view(out.size(0), -1).mean(dim=1)
        probs = torch.sigmoid(out)
        expected_evals = prob_to_pawns(probs).tolist()

    # Инвертируем оценку обратно для интерфейса (если ход черных)
    for i, fen in enumerate(fens):
        val = expected_evals[i]
        if ' b ' in fen:
            expected_evals[i] = -val
        else:
            expected_evals[i] = val

    return expected_evals

def evaluate_model_metrics(model: torch.nn.Module, dataloader: torch.utils.data.DataLoader, device: torch.device) -> tuple[float, float]:
    model.eval()
    total_mae_pawns = 0.0
    correct_signs = 0
    total_positions = 0

    eval_pbar = tqdm(dataloader, desc="Testowanie modelu", leave=True)

    with torch.no_grad():
        for val_data in eval_pbar:
            boards, turns, castling, eps, batch_targets = [x.to(device, non_blocking=True) for x in val_data]
            
            batch_inputs = build_batch_on_gpu(boards, turns, castling, eps)

            batch_targets_canonical = torch.where(turns == 0, 1.0 - batch_targets, batch_targets)

            out = model(batch_inputs)
            if out.ndim > 1:
                out = out.view(out.size(0), -1).mean(dim=1)

            probs = torch.sigmoid(out)

            preds_pawns = prob_to_pawns(probs)
            targets_pawns = prob_to_pawns(batch_targets_canonical)

            # ensure both are 1D tensors of shape (batch,)
            preds_pawns = preds_pawns.view(-1)
            targets_pawns = targets_pawns.view(-1)

            mae = torch.abs(preds_pawns - targets_pawns).sum().item()
            total_mae_pawns += mae

            preds_signs = torch.sign(preds_pawns)
            targets_signs = torch.sign(targets_pawns)
            correct_signs += (preds_signs == targets_signs).sum().item()

            total_positions += batch_targets.size(0)

            running_mae = total_mae_pawns / total_positions
            running_acc = (correct_signs / total_positions) * 100.0
            eval_pbar.set_postfix({'MAE': f"{running_mae:.2f}", 'Acc': f"{running_acc:.1f}%"})

    avg_mae_pawns = total_mae_pawns / total_positions
    sign_accuracy = (correct_signs / total_positions) * 100.0

    print(f"\n📊 Wyniki testowania ({total_positions} pozycji):")
    print(f"\tOcena przewagi (Sign Accuracy): {sign_accuracy:.1f}%")
    print(f"\tŚredni błąd (MAE):          {avg_mae_pawns:.2f} piona")

    return avg_mae_pawns, sign_accuracy

def show_model_stats(model, val_loader, df):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    test_positions = [
        [df["FEN"].iloc[i], df["Evaluation"].iloc[i]] for i in range(1000)
    ]
    
    just_fens = [item[0] for item in test_positions]
    evaluations = predict_evaluation(model, just_fens)

    # Списки для подсчета средней ошибки по диапазонам
    # Бины: symmetric ranges specified by user: 1,2,3,5,7,10
    bins = [0, 1, 2, 3, 5, 7, 10]
    # prepare container for each bin (center, then increasing rings)
    bin_diffs = {i: [] for i in range(len(bins)-1)}

    print("\nWyniki oceny przez sieć neuronową:")
    for [fen, correct_eval], eval_score in zip(test_positions, evaluations):
        eval_str = str(correct_eval).strip()
        if eval_str.startswith('#'):
            true_pawns = 10.0 if eval_str[1] == '+' else -10.0
        else:
            true_pawns = float(eval_str) / 100.0
            
        diff = abs(eval_score - true_pawns)

        # Распределяем ошибку по симметричным корзинам на основе абсолютной истинной оценки
        a = abs(true_pawns)
        # Найдём bin: first bin includes [0..1], next bins are (1..2], (2..3], (3..5], (5..7], (7..10]
        for i in range(len(bins)-1):
            low = bins[i]
            high = bins[i+1]
            if i == 0:
                if a <= high:
                    bin_diffs[i].append(diff)
                    break
            else:
                if low < a <= high:
                    bin_diffs[i].append(diff)
                    break
            
        short_fen = fen.split()[0]
        if len(short_fen) > 30:
            short_fen = short_fen[:27] + "..."
            
        # print(f"FEN: {short_fen:<30} | Model: {eval_score:>6.2f} | Prawdziwa: {true_pawns:>6.2f} | Błąd: {diff:>5.2f} piona")

    # === ВЫВОД РАЗНИЦЫ ПО ДИАПАЗОНАМ ===
    print("\n" + "="*75)
    print("📊 Statystyki szczegółowe (na podstawie 1000 pozycji):")

    # Формат вывода для каждой пары симметричных диапазонов
    labels = ["[-1..1]", "(1..2]", "(2..3]", "(3..5]", "(5..7]", "(7..10]"]
    for i, label in enumerate(labels):
        vals = bin_diffs.get(i, [])
        if vals:
            avg = sum(vals) / len(vals)
            print(f"  {label:8}  Średni błąd: {avg:>5.2f} piona  (Pozycji: {len(vals)})")
        else:
            print(f"  {label:8}  Brak pozycji (Pozycji: 0)")

    print("="*75 + "\n")

    evaluate_model_metrics(model, val_loader, device)
