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

    # Tablica wartości dla każdego koszyka od -10 do +10
    bucket_values = torch.linspace(-10.0, 10.0, steps=100, device=device)

    with torch.no_grad():
        # Pobieramy surowe logity
        logits = model(inputs)
        # Zamieniamy logity na rzeczywiste procenty prawdopodobieństwa (suma = 1.0)
        probabilities = torch.softmax(logits, dim=-1)
        
        # Wartość oczekiwana: Prawdopodobieństwo * Wartość koszyka
        # Daje to idealną ułamkową precyzję (np. 1.27 piona)
        expected_evals = torch.sum(probabilities * bucket_values, dim=-1).tolist()

    return expected_evals

def evaluate_model_metrics(model: torch.nn.Module, dataloader: torch.utils.data.DataLoader, device: torch.device) -> tuple[float, float]:
    model.eval()
    total_mae_pawns = 0.0
    correct_signs = 0
    total_positions = 0
    bucket_values = torch.linspace(-10.0, 10.0, steps=100, device=device)

    # Оборачиваем dataloader в tqdm
    eval_pbar = tqdm(dataloader, desc="Testowanie modelu", leave=True)

    with torch.no_grad():
        for val_data in eval_pbar:
            boards, turns, castling, eps, batch_targets = [x.to(device, non_blocking=True) for x in val_data]
            
            # Budujemy batch na GPU
            batch_inputs = build_batch_on_gpu(boards, turns, castling, eps)

            # Przekazujemy do modelu
            logits = model(batch_inputs) 

            probabilities = torch.softmax(logits, dim=-1)
            preds_pawns = torch.sum(probabilities * bucket_values, dim=-1) 
            targets_pawns = bucket_values[batch_targets] 

            mae = torch.abs(preds_pawns - targets_pawns).sum().item()
            total_mae_pawns += mae

            preds_signs = torch.sign(preds_pawns)
            targets_signs = torch.sign(targets_pawns)
            correct_signs += (preds_signs == targets_signs).sum().item()

            total_positions += batch_targets.size(0)

            # Вычисляем текущие метрики на лету и выводим их в прогресс-бар
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
        [df["FEN"].iloc[i], df["Evaluation"].iloc[i]] for i in range(10)
    ]
    
    # 1. Wyciągamy tylko ciągi FEN dla sieci neuronowej
    just_fens = [item[0] for item in test_positions]

    # 2. Wykonujemy predykcję
    evaluations = predict_evaluation(model, just_fens)

    # 3. Wyświetlamy wynik
    print("\nWyniki oceny przez sieć neuronową:")
    for [fen, correct_eval], eval_score in zip(test_positions, evaluations):
        print(f"\nPozycja: {fen.split()[0]}")
        print(f"Ocena modelu: {eval_score:.2f} piona (lub {int(eval_score * 100)} centypionach) | Prawdziwa: {correct_eval}")

    # STATYSTYKA
    evaluate_model_metrics(model, val_loader, device)
