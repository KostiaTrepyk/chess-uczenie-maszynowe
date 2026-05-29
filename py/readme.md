# Szachy — Ocena Pozycji Szachowych za Pomocą Sieci Neuronowej

Projekt implementuje hybrydową sieć neuronową (CNN opartą na architekturze ResNet połączoną z modułem Transformer) do przewidywania oceny pozycji szachowych bezpośrednio na podstawie notacji FEN. Model klasyfikuje pozycję do jednego ze 100 koszyków (od -10 do +10 pionów), zapewniając optymalny balans między dokładnością a wydajnością.

## Zbiór Danych

Projekt wykorzystuje zbiór danych z ocenami milionów pozycji szachowych wygenerowanych przez silnik Stockfish:
* **Źródło Kaggle:** [Chess Evaluations](https://www.kaggle.com/datasets/ronakbadhe/chess-evaluations)
* *Uwaga:* Pobrany plik `chessData.csv` należy umieścić w głównym katalogu projektu przed uruchomieniem skryptów.

## Architektura Modelu

W celu maksymalizacji jakości predykcji połączono analizę przestrzenną z globalnym kontekstem:

* **Reprezentacja wejściowa:** Zoptymalizowany tensor 15x8x8 (12 warstw figur, kolej ruchu, roszady, bicie w przelocie).
* **Ekstrakcja cech (CNN):** 10 bloków *ResidualBlock* z mechanizmem *Squeeze-and-Excitation* (skupienie uwagi sieci na najważniejszych polach).
* **Globalny kontekst:** *TransformerBlock* z mechanizmem Multihead Attention (4 głowy), analizujący relacje między oddalonymi od siebie figurami na całej szachownicy.

## Struktura Projektu

| Plik | Opis |
| :--- | :--- |
| `dataset.py` | Szybka tokenizacja FEN i zarządzanie pamięcią. Pakuje dane do `FastChessDataset` (w RAM jako `uint8`). |
| `model.py` | Definicja architektury (ResNet + Transformer), budowa batchy bezpośrednio na GPU oraz pętla ucząca z automatyczną mieszaną precyzją (AMP). |
| `train_model.py` | Główny skrypt uruchamiający proces trenowania modelu. Zapisuje najlepszą wersję jako `best_chess_model.pth`. |
| `model_stats.py` | Skrypt testowy walidujący model. Wyświetla metryki (MAE, dokładność znaku przewagi) na przykładowych pozycjach z datasetu. |
| `export_model.py` | Eksportuje nauczony model z PyTorch do uniwersalnego formatu `chess_model.onnx` (gotowy do wdrożenia, np. w Unity). |

## Instalacja Zależności

Zalecane jest użycie środowiska wirtualnego (venv/conda) ze względu na rozmiar bibliotek.

```bash
# Instalacja wymaganych pakietów pomocniczych
pip install chess tqdm onnx onnxscript pandas

# Instalacja PyTorch z obsługą akceleracji GPU (CUDA 12.6)
pip install torch --index-url [https://download.pytorch.org/whl/cu126](https://download.pytorch.org/whl/cu126)
```

## Przykład aktywacji środowiska wirtualnego na Windows (PowerShell/CMD)

```bash
cd ./py/venv/Scripts
activate
cd ../..
```

## Uruchamianie Projektu

### Aby rozpocząć pracę z projektem, użyj poniższych komend w głównym katalogu:

1. Uruchomienie treningu modelu (wymaga pliku chessData.csv):

```bash
python train_model.py
```

2. Testowanie modelu i wyświetlanie statystyk (wymaga wytrenowanego pliku .pth):

```bash
python model_stats.py
```

3. Eksport do formatu ONNX (do środowisk produkcyjnych / C#):

```bash
python export_model.py
```
