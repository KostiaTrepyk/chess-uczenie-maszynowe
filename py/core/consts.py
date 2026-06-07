NUM_BLOCKS = 20 # 10 20
CHANNELS = 256
TransformerBlockHeads = 16

# 256 512
BATCH_SIZE = 256
MAX_VALIDATION_BATCHES = 100

CHECKS_PER_EPOCH = 40
EPOCHS = 3

START_LR = 0.0003
RESUME_START_LR = 0.000025

KERNEL_SIZE = 3
PADDING = 1

PIECE_TO_CHANNEL = {
    'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5,
    'p': 6, 'n': 7, 'b': 8, 'r': 9, 'q': 10, 'k': 11
}

CHAR_TO_INT_TOKEN = {
    'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5,
    'p': 6, 'n': 7, 'b': 8, 'r': 9, 'q': 10, 'k': 11
}
