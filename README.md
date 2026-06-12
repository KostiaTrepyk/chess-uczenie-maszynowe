# ♟️ Chess AI Engine (ResNet + Transformer)

A hybrid chess AI engine combining Deep Learning for position evaluation with classic search algorithms (Alpha-Beta Pruning) for tactical decision-making. The project consists of a **PyTorch** machine learning pipeline and a high-performance **Next.js** web interface where the engine's logic is executed.

## 📸 Screenshots

<p align="center">
  <img width="40%" alt="Chess AI Interface" src="https://github.com/user-attachments/assets/afbd0f3e-51bc-4d49-9656-5a22f85f6191" />
  <img width="59%"  alt="Engine Evaluation" src="https://github.com/user-attachments/assets/8f89515b-7c6e-40b2-b680-001fec0c572b" />
</p>

## ✨ Key Features

- **Hybrid Neural Network (PyTorch):** Position evaluation using a custom architecture featuring 10 ResNet blocks and 3 Transformer (Multi-Head Attention) blocks.
- **Two-Headed Output:** The model simultaneously predicts positional advantage (in pawns) and checkmate probability.
- **Advanced Search Algorithm (TypeScript):**
  - **Alpha-Beta Pruning:** Efficiently prunes unpromising branches in the decision tree.
  - **Quiescence Search:** Forces the calculation of all capture sequences at the end of the search depth to prevent the horizon effect (blundering pieces during exchanges).
  - **Smart Move Ordering (MVV-LVA):** Implements the *Most Valuable Victim - Least Valuable Attacker* heuristic to drastically speed up the Alpha-Beta search.
- **Anti-Repetition Mechanism:** Built-in logic to avoid threefold repetition draws by dynamically selecting the next best alternative move.
- **Server-Side Inference (ONNX):** Utilizes `onnxruntime-node` for multi-threaded, low-latency neural network execution directly on the CPU.

## 🛠 Tech Stack

**Machine Learning (Backend / Training):**
- Python 3
- PyTorch (Model training, OneCycleLR, JIT Compilation)
- ONNX (Model export)
- `python-chess` (Rules validation and dataset processing)

**Web Application (Frontend / Engine AI):**
- React 19 / Next.js 16
- TypeScript
- Tailwind CSS
- `chess.js` (Move validation and game state)
- `react-chessboard` (Interactive UI)
- `onnxruntime-node` (AI inference environment)

## 📂 Project Structure

```text
chess-uczenie-maszynowe/
├── py/                     # ML Pipeline
│   ├── core/               # Network architecture (architecture.py) and training logic
│   ├── train_model.py      # Training script entry point
│   └── export_model.py     # Script to convert .pth weights to .onnx format
│
└── webapp/                 # Next.js Application
    ├── src/app/api/move/   # AI Engine API endpoint (Alpha-Beta logic)
    ├── src/lib/            # Tensor utilities and ONNX session initialization
    ├── src/app/page.tsx    # Game interface (PvP, AI vs Human, AI vs AI)
    └── models/             # Directory for generated .onnx models
```

## 🚀 Installation & Setup
**1. Model Preparation (Python)**
If you want to train the model from scratch or export an existing one:

```bash
cd py

# Create a virtual environment and install dependencies
python -m venv venv

# For Windows:
venv\Scripts\activate
# For Unix/macOS:
# source venv/bin/activate

pip install torch onnx python-chess numpy tqdm

# Run training (requires .csv datasets)
python train_model.py

# Export the trained model (automatically copies files to webapp/models)
python export_model.py
```

**2. Running the Web App (Next.js)**
Ensure that chess_model.onnx and chess_model.onnx.data (if the model exceeds 2GB) are located in the webapp/models/ directory.

```bash
cd webapp
npm install

# Start the local development server
npm run dev
```
Open http://localhost:3000 in your browser. You can select the game mode (Play as White, Black, PvP, or Model vs Model) and adjust the AI search depth.

## 🧠 How the AI Thinks

1. Move Generation: The server receives the current FEN from the client and generates legal moves using chess.js.
2. Move Ordering: Moves are sorted (checks and MVV-LVA captures first) to maximize Alpha-Beta pruning efficiency.
3. Tree Search: The algorithm explores the game tree up to the specified Depth. Once the depth limit is reached, Quiescence Search takes over to resolve any pending captures.
4. Evaluation (Tensors): At the leaf nodes, the FEN string is converted into a 15-channel tensor (piece positions, castling rights, turn) and fed into the ONNX model, which returns the final static evaluation score.

## 📄 License
MIT License.
