import torch
import torch.nn as nn
import torch.nn.functional as F

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim=None, dropout=0.0):
        super().__init__()
        
        hidden_dim = hidden_dim or dim * 4
        c = (6 / (dim + hidden_dim))**0.5

        fc1 = nn.Linear(dim, hidden_dim)
        nn.init.uniform_(fc1.weight, -c, c)

        act = nn.ReLU()

        fc2 = nn.Linear(hidden_dim, dim)
        nn.init.uniform_(fc2.weight, -c, c)

        drop = nn.Dropout(dropout)
        self.FF = nn.Sequential(fc1, act, fc2, drop)

    def forward(self, x):
        return self.FF(x)

class SEBlock(nn.Module):
    """SENet module as in the paper, applied after CAFF fusion."""
    def __init__(self, c, reduction=8):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(c, c // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(c // reduction, c, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.shape
        w = self.pool(x).view(b, c)
        w = self.fc(w).view(b, c, 1, 1)
        return x * w

class CAFF(nn.Module):
    """
    Cross-Attention Feature Fusion (CAFF) Module
    Implements the version described in the 4DRadDet paper:
        - Q from Cluster Enhancement Branch (FD_BEV)
        - K,V from Pillar Enhancement Branch (FP_BEV)
    No conv-based embedding — just flatten + linear projection + positional encoding.
    """
    def __init__(self, channels=64, nheads=8, dropout=0.0, use_se=True):
        super().__init__()
        self.C = channels
        self.nheads = nheads

        # Linear projections for Q, K, V
        c = (6 / (channels + channels))**0.5
        self.WQ = nn.Linear(channels, channels)
        nn.init.uniform_(self.WQ.weight, -c, c)
        self.WK = nn.Linear(channels, channels)
        nn.init.uniform_(self.WK.weight, -c, c)
        self.WV = nn.Linear(channels, channels)
        nn.init.uniform_(self.WV.weight, -c, c)
        self.WO = nn.Linear(channels, channels)
        nn.init.uniform_(self.WO.weight, -c, c)

        # Feed Forward block (FFN)
        self.ffn = FeedForward(channels, dropout=dropout)

        # Learnable residual scaling parameters δ₁–δ₄ (initialized to 1)
        self.delta1 = nn.Parameter(torch.ones(1))
        self.delta2 = nn.Parameter(torch.ones(1))
        self.delta3 = nn.Parameter(torch.ones(1))
        self.delta4 = nn.Parameter(torch.ones(1))

        # Learnable positional encodings
        self.pos_pillar = None
        self.pos_cluster = None

        # Optional SENet + channel fusion (as in paper)
        self.use_se = use_se
        if use_se:
            self.se = SEBlock(channels)

        self.fuse = nn.Sequential(
            nn.Conv2d(2 * channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )

        self.dropout = nn.Dropout(dropout)

    def _make_pos_encoding(self, H, W, device):
        """Create or update learnable positional encodings of shape (HW, 1, C)."""
        if (self.pos_pillar is None) or (self.pos_pillar.shape[0] != H * W):
            self.pos_pillar = nn.Parameter(torch.empty(H * W, 1, self.C, device=device))
            self.pos_cluster = nn.Parameter(torch.empty(H * W, 1, self.C, device=device))
            nn.init.trunc_normal_(self.pos_pillar, std=0.02)  
            nn.init.trunc_normal_(self.pos_cluster, std=0.02)

    def forward(self, FP_BEV, FD_BEV):
        """
        Args:
            FP_BEV: Pillar branch BEV features (B, C, H, W) -> K,V
            FD_BEV: Cluster branch BEV features (B, C, H, W) -> Q
        Returns:
            F_fused: (B, C, H, W)
        """
        B, C, H, W = FP_BEV.shape
        device = FP_BEV.device
        self._make_pos_encoding(H, W, device)

        # Flatten BEV maps into token sequences (HW, B, C)
        TP = FP_BEV.flatten(2).permute(2, 0, 1)  # (HW, B, C)
        TD = FD_BEV.flatten(2).permute(2, 0, 1)  # (HW, B, C)

        # Add positional encodings
        TP = TP + self.pos_pillar
        TD = TD + self.pos_cluster

        # Linear projections
        K = self.WK(TP)
        V = self.WV(TP)
        Q = self.WQ(TD)

        # Compute scaled dot-product attention manually
        dk = Q.shape[-1] // self.nheads
        Q_ = Q.view(H*W, B, self.nheads, dk).transpose(0, 1)  # (B, HW, heads, dk)
        K_ = K.view(H*W, B, self.nheads, dk).transpose(0, 1)
        V_ = V.view(H*W, B, self.nheads, dk).transpose(0, 1)

        attn_scores = torch.einsum('bqhd,bkhd->bhqk', Q_, K_) / (dk ** 0.5)  # q,k = HW
        attn = F.softmax(attn_scores, dim=-1)
        Z = torch.einsum('bhqk,bkhd->bqhd', attn, V_)  # -> (B, HW, h, dk)

        # Merge heads and revert shape
        Z = Z.reshape(B, H*W, self.C).permute(1, 0, 2)  # (HW, B, C)

        # Z' = δ₁·TP + δ₂·(Z·WO)
        Z_proj = self.WO(Z)
        Z_prime = self.delta1 * TP + self.delta2 * Z_proj

        # TPD = δ₃·Z' + δ₄·FFN(Z')
        Z_ffn = self.ffn(Z_prime)
        TPD = self.delta3 * Z_prime + self.delta4 * Z_ffn

        # Reshape back to (B, C, H, W)
        FPD_BEV = TPD.permute(1, 2, 0).view(B, C, H, W)

        #Add FP_BEV residual
        FPD_BEV = FPD_BEV + FP_BEV

        # Optional SENet and fusion
        if self.use_se:
            FPD_BEV = self.se(FPD_BEV)

        F_fused = self.fuse(torch.cat([FP_BEV, FPD_BEV], dim=1))
        return F_fused