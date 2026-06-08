import torch
import torch.nn as nn

from core.consts import NUM_BLOCKS, CHANNELS, KERNEL_SIZE, PADDING, TransformerBlockHeads

class TransformerBlock(nn.Module):
    def __init__(self, channels: int, heads: int = 16):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim=channels, num_heads=heads, batch_first=True)
        self.norm1 = nn.LayerNorm(channels)
        self.norm2 = nn.LayerNorm(channels)
        self.mlp = nn.Sequential(
            nn.Linear(channels, channels * 2),
            nn.GELU(),
            nn.Linear(channels * 2, channels)
        )
        self.pos_embedding = nn.Parameter(torch.randn(1, 64, channels) * 0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.size()
        x_flat = x.view(b, c, h * w).permute(0, 2, 1)
        x_flat = x_flat + self.pos_embedding
        norm_x1 = self.norm1(x_flat)
        attn_out, _ = self.attention(norm_x1, norm_x1, norm_x1)
        x_flat = x_flat + attn_out
        norm_x2 = self.norm2(x_flat)
        mlp_out = self.mlp(norm_x2)
        out = x_flat + mlp_out
        return out.permute(0, 2, 1).view(b, c, h, w)

class ResidualBlock(nn.Module):
    def __init__(self, channels: int = 256):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.act = nn.SiLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.se = SEBlock(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)
        out.add_(residual)
        out = self.act(out)
        return out

class ChessResNet(nn.Module):
    def __init__(self, num_blocks=NUM_BLOCKS, channels=CHANNELS, num_transformer_blocks: int = 3):
        super().__init__()
        self.input_conv = nn.Sequential(
            nn.Conv2d(15, channels, kernel_size=KERNEL_SIZE, padding=PADDING),
            nn.BatchNorm2d(channels),
            nn.LeakyReLU(0.1)
        )
        self.resnet_blocks = nn.Sequential(*[ResidualBlock(channels) for _ in range(num_blocks)])
        self.transformer = nn.Sequential(*[TransformerBlock(channels=channels, heads=TransformerBlockHeads) for _ in range(num_transformer_blocks)])
        
        # Общие признаки для оценки (расширяем выход до 512)
        self.value_features = nn.Sequential(
            nn.Conv2d(channels, 4, kernel_size=1),
            nn.BatchNorm2d(4),
            nn.LeakyReLU(0.1),
            nn.Flatten(),
            nn.Linear(256, 512), # <--- Немного расширили горлышко
            nn.LeakyReLU(0.1),
        )
        
        # --- НОВАЯ АРХИТЕКТУРА ГОЛОВ ---
        # Голова 1: Точная оценка пешек (больше слоев)
        self.score_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.LeakyReLU(0.1),
            nn.Linear(256, 1)
        )
        
        # Голова 2: Вероятность мата (меньше слоев)
        self.mate_head = nn.Sequential(
            nn.Linear(512, 128),
            nn.LeakyReLU(0.1),
            nn.Linear(128, 1)
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.input_conv(x)
        x = self.resnet_blocks(x)
        x = self.transformer(x)
        
        features = self.value_features(x)
        
        # Пропускаем через изолированные слои
        return self.score_head(features).squeeze(-1), self.mate_head(features).squeeze(-1)

def load_model(num_blocks=NUM_BLOCKS, channels=CHANNELS)-> ChessResNet:
    # Wczytujemy gotowy model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 2. Tworzymy pustą architekturę sieci i przenosimy na urządzenie
    model = ChessResNet(num_blocks, channels).to(device)

    # 3. Wczytujemy wagi z pliku
    # map_location gwarantuje poprawne wczytanie, nawet jeśli model był trenowany na GPU a uruchamiany na CPU
    print(f'Wczytywanie modelu z pliku: best_chess_model.pth')
    model.load_state_dict(torch.load("best_chess_model.pth", map_location=device))

    # 4. Koniecznie przełączzujemy na tryb predykcji (eval)
    model.eval()

    return model

class SEBlock(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super(SEBlock, self).__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.squeeze(x)
        y = self.excitation(y)
        return x * y
