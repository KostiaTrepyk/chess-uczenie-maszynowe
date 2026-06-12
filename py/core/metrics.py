import torch
import torch.nn as nn
from tqdm import tqdm
from typing import List

from core.features import fen_to_tensor, build_batch_on_gpu

def predict_evaluation(model: nn.Module, fens: List[str]) -> List[float]:
    if not fens:
        return []

    model.eval()
    device = next(model.parameters()).device
    inputs = torch.stack([fen_to_tensor(fen) for fen in fens]).to(device)

    with torch.no_grad():
        score_preds, mate_logits = model(inputs)
        
        score_preds = score_preds.view(-1)
        mate_logits = mate_logits.view(-1)
        
        # Переводим логиты в вероятность (от 0 до 1)
        mate_probs = torch.sigmoid(mate_logits)
        
        expected_evals = []
        for i in range(len(fens)):
            raw_score = score_preds[i].item()
            mate_prob = mate_probs[i].item()
            
            # --- ЛОГИКА ДВИЖКА (ЭФФЕКТ КОМПАСА) ---
            if mate_prob > 0.75:
                # Игнорируем сырые пешки, если сеть уверена в мате
                mate_score = 20.0 + mate_prob * 10.0
                final_score = mate_score if raw_score > 0 else -mate_score
            else:
                # Обрезаем галлюцинации в обычной игре
                final_score = max(-15.0, min(15.0, raw_score))
                
            expected_evals.append(final_score)

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
    
    # Счетчики для Матрицы Ошибок Мата
    true_positives = 0
    true_negatives = 0
    false_positives = 0
    false_negatives = 0

    eval_pbar = tqdm(dataloader, desc="Testowanie modelu", leave=True)

    with torch.no_grad():
        for val_data in eval_pbar:
            boards, turns, castling, eps, batch_targets = [x.to(device, non_blocking=True) for x in val_data]
            
            batch_inputs = build_batch_on_gpu(boards, turns, castling, eps)

            # Пешки инвертируются через минус (канонический вид)
            batch_targets_canonical = torch.where(turns == 0, -batch_targets, batch_targets)

            score_preds, mate_logits = model(batch_inputs)

            preds_pawns = score_preds.view(-1)
            targets_pawns = batch_targets_canonical.view(-1)
            mate_probs = torch.sigmoid(mate_logits.view(-1))

            # --- ПРИМЕНЯЕМ КОМБИНИРОВАННУЮ ЛОГИКУ К ТЕНЗОРАМ ---
            # 1. Обрезаем обычные оценки до 15 пешек
            final_preds = torch.clamp(preds_pawns, -15.0, 15.0)
            
            # 2. Находим матовые позиции (вероятность > 0.75)
            mate_mask = mate_probs > 0.75
            mate_scores = 20.0 + mate_probs * 10.0
            
            # 3. Применяем матовую оценку там, где сработала маска
            final_preds = torch.where(
                mate_mask, 
                torch.where(preds_pawns > 0, mate_scores, -mate_scores), 
                final_preds
            )

            # Считаем MAE уже по финальным, исправленным предсказаниям
            mae = torch.abs(final_preds - targets_pawns).sum().item()
            total_mae_pawns += mae

            preds_signs = torch.sign(final_preds)
            targets_signs = torch.sign(targets_pawns)
            correct_signs += (preds_signs == targets_signs).sum().item()

            # --- ОЦЕНКА МАТОВОЙ ГОЛОВЫ (Classification) ---
            # Допустим, в датасете мат - это когда |target| >= 20.0
            target_is_mate = torch.abs(targets_pawns) >= 20.0
            pred_is_mate = mate_mask
            
            true_positives += (pred_is_mate & target_is_mate).sum().item()
            true_negatives += (~pred_is_mate & ~target_is_mate).sum().item()
            false_positives += (pred_is_mate & ~target_is_mate).sum().item()
            false_negatives += (~pred_is_mate & target_is_mate).sum().item()

            total_positions += batch_targets.size(0)

            running_mae = total_mae_pawns / total_positions
            running_acc = (correct_signs / total_positions) * 100.0
            eval_pbar.set_postfix({'MAE': f"{running_mae:.2f}", 'Acc': f"{running_acc:.1f}%"})

    avg_mae_pawns = total_mae_pawns / total_positions
    sign_accuracy = (correct_signs / total_positions) * 100.0

    print(f"\n📊 Wyniki testowania ({total_positions} pozycji):")
    print(f"\tOcena przewagi (Sign Accuracy): {sign_accuracy:.1f}%")
    print(f"\tŚredni błąd (Skorygowane MAE):  {avg_mae_pawns:.2f} piona")
    
    print("\n🧠 Klasyfikacja Mata (Confusion Matrix):")
    print(f"\tPrawdziwe Negatywy (Zwykłe pozycje zgadnięte): {true_negatives}")
    print(f"\tFałszywe Pozytywy (Fałszywy alarm mata):     {false_positives}  <-- Gwałtowne pomyłki!")
    print(f"\tFałszywe Negatywy (Pominięte maty):          {false_negatives}  <-- Ważne do poprawy")
    print(f"\tPrawdziwe Pozytywy (Mat zgadnięty poprawnie): {true_positives}")

    return avg_mae_pawns, sign_accuracy

def show_model_stats(model, val_loader, df):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    test_positions = [
        [df["FEN"].iloc[i], df["Evaluation"].iloc[i]] for i in range(10000)
    ]
    
    just_fens = [item[0] for item in test_positions]
    # Теперь predict_evaluation использует обе головы!
    evaluations = predict_evaluation(model, just_fens)

    bins = [0, 2, 5, 7, 10, 15, 100]
    bin_diffs = {i: [] for i in range(len(bins)-1)}

    print("\nWyniki oceny przez sieć neuronową:")
    for [fen, correct_eval], eval_score in zip(test_positions, evaluations):
        eval_str = str(correct_eval).strip()
        if eval_str.startswith('#'):
            true_pawns = 30.0 if eval_str[1] == '+' else -30.0
        else:
            true_pawns = float(eval_str) / 100.0
            
        diff = abs(eval_score - true_pawns)

        a = abs(true_pawns)
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

    print("\n" + "="*75)
    print("📊 Statystyki szczegółowe (na podstawie 10000 pozycji):")

    labels = ["[-2..2]", "(2..5]", "(5..7]", "(7..10]", "(10..15]", "(15..100]"]
    for i, label in enumerate(labels):
        vals = bin_diffs.get(i, [])
        if vals:
            avg = sum(vals) / len(vals)
            print(f"  {label:8}  Średni błąd: {avg:>5.2f} piona  (Pozycji: {len(vals)})")
        else:
            print(f"  {label:8}  Brak pozycji (Pozycji: 0)")

    print("="*75 + "\n")
    evaluate_model_metrics(model, val_loader, device)
