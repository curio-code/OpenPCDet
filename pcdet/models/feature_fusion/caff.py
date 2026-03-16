import math
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
    def __init__(self, model_cfg=None, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg

        if self.model_cfg is not None and (
            hasattr(self.model_cfg, 'CHANNELS') or hasattr(self.model_cfg, 'EMBED_DIM')
        ):
            channels = getattr(self.model_cfg, 'CHANNELS')
            nheads = getattr(self.model_cfg, 'HEADS', getattr(self.model_cfg, 'NUM_HEADS', kwargs.get('nheads', 1)))
            dropout = getattr(self.model_cfg, 'DROPOUT', getattr(self.model_cfg, 'DROP', kwargs.get('dropout', 0.0)))
            use_se = getattr(self.model_cfg, 'USE_SE', getattr(self.model_cfg, 'USE_SENET', kwargs.get('use_se', False)))
            se_reduction = getattr(self.model_cfg, 'SE_REDUCTION', getattr(self.model_cfg, 'SENET_REDUCTION', kwargs.get('se_reduction', 8)))
        else:
            channels = kwargs.get('channels', 64)
            nheads = kwargs.get('nheads', 8)
            dropout = kwargs.get('dropout', 0.0)
            use_se = kwargs.get('use_se', True)
            se_reduction = kwargs.get('se_reduction', 8)

        if channels is None:
            raise ValueError("CAFF must receive the BEV feature dimension via config or constructor arguments.")

        self.C = channels
        self.nheads = nheads
        channels = self.C

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
            self.se = SEBlock(channels, reduction=se_reduction)

        self.fuse = nn.Sequential(
            nn.Conv2d(2 * channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )

        self.dropout = nn.Dropout(dropout)
        self.pillar_proj = None
        self.cluster_proj = None

    def _make_pos_encoding(self, H, W, device):
        """Create or update learnable positional encodings of shape (HW, 1, C)."""
        if (self.pos_pillar is None) or (self.pos_pillar.shape[0] != H * W):
            self.pos_pillar = nn.Parameter(torch.empty(H * W, 1, self.C, device=device))
            self.pos_cluster = nn.Parameter(torch.empty(H * W, 1, self.C, device=device))
            nn.init.trunc_normal_(self.pos_pillar, std=0.02)  
            nn.init.trunc_normal_(self.pos_cluster, std=0.02)

    def forward(self, *args):
        """
        Supports two calling patterns:
            1. forward(batch_dict) where batch_dict carries the required tensors and will be updated in-place.
            2. forward(FP_BEV, FD_BEV) returning the fused tensor directly.
        """
        if len(args) == 1 and isinstance(args[0], dict):
            batch_dict = args[0]
            FP_BEV = batch_dict.get('spatial_features', None)
            FD_BEV = batch_dict.get('spatial_features_ceb', None)
            assert FP_BEV is not None, "CAFF expects 'spatial_features' from PEB in batch_dict."
            assert FD_BEV is not None, "CAFF expects 'spatial_features_ceb' from CEB in batch_dict."

            fused = self._fuse(FP_BEV, FD_BEV)
            batch_dict['spatial_features_peb'] = FP_BEV
            batch_dict['spatial_features_fused'] = fused
            batch_dict['spatial_features'] = fused
            return batch_dict

        if len(args) != 2:
            raise ValueError("CAFF.forward expects either (batch_dict,) or (FP_BEV, FD_BEV).")

        FP_BEV, FD_BEV = args
        return self._fuse(FP_BEV, FD_BEV)

    def _fuse(self, FP_BEV, FD_BEV):
        """
        Args:
            FP_BEV: Pillar branch BEV features (B, C, H, W) -> K,V
            FD_BEV: Cluster branch BEV features (B, C, H, W) -> Q
        Returns:
            F_fused: (B, C, H, W)
        """
        if FP_BEV.shape[1] != self.C:
            if (self.pillar_proj is None) or (self.pillar_proj.in_channels != FP_BEV.shape[1]):
                self.pillar_proj = nn.Conv2d(FP_BEV.shape[1], self.C, kernel_size=1, bias=False)
                self.pillar_proj = self.pillar_proj.to(device=FP_BEV.device, dtype=FP_BEV.dtype)
                nn.init.kaiming_uniform_(self.pillar_proj.weight, a=math.sqrt(5))
            FP_BEV = self.pillar_proj(FP_BEV)

        if FD_BEV.shape[1] != self.C:
            if (self.cluster_proj is None) or (self.cluster_proj.in_channels != FD_BEV.shape[1]):
                self.cluster_proj = nn.Conv2d(FD_BEV.shape[1], self.C, kernel_size=1, bias=False)
                self.cluster_proj = self.cluster_proj.to(device=FD_BEV.device, dtype=FD_BEV.dtype)
                nn.init.kaiming_uniform_(self.cluster_proj.weight, a=math.sqrt(5))
            FD_BEV = self.cluster_proj(FD_BEV)

        B, C, H, W = FP_BEV.shape
        assert self.C % self.nheads == 0, "CHANNELS must be divisible by number of HEADS."
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
