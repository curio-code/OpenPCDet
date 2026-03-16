import pytest
easydict = pytest.importorskip('easydict')
from easydict import EasyDict

from pcdet.models.backbones_3d.cfe import CrossAttentionFeatureFusion

torch = pytest.importorskip('torch')

def test_caff_forward_shapes_and_state():
    cfg = EasyDict({
        'EMBED_DIM': 64,
        'NUM_HEADS': 8,
        'FFN_EXPANSION': 4.0,
        'USE_SENET': True,
        'SENET_REDUCTION': 16,
        'DROPOUT': 0.1,
    })
    module = CrossAttentionFeatureFusion(cfg)

    B, C_peb, C_ceb, H, W = 2, 48, 32, 6, 5
    torch.manual_seed(0)
    batch_dict = {
        'spatial_features': torch.randn(B, C_peb, H, W),
        'spatial_features_ceb': torch.randn(B, C_ceb, H, W),
    }

    out = module(batch_dict)
    assert 'spatial_features' in out and 'spatial_features_fused' in out
    fused = out['spatial_features']
    assert fused.shape == (B, cfg.EMBED_DIM, H, W)
    assert torch.all(torch.isfinite(fused))
    assert torch.equal(fused, out['spatial_features_fused'])
    assert 'spatial_features_peb' in out

    assert module.patch_embed_p is not None
    assert module.patch_embed_d is not None
    assert module.pos_embed is not None
    assert module.pos_embed.shape[1] == H * W

    for delta in (module.delta1, module.delta2, module.delta3, module.delta4):
        assert delta.detach().item() == pytest.approx(1.0, rel=1e-5)


def test_caff_raises_on_spatial_mismatch():
    cfg = EasyDict({'EMBED_DIM': 32, 'NUM_HEADS': 8})
    module = CrossAttentionFeatureFusion(cfg)

    batch_dict = {
        'spatial_features': torch.randn(1, 32, 4, 4),
        'spatial_features_ceb': torch.randn(1, 32, 5, 4),
    }

    with pytest.raises(AssertionError):
        module(batch_dict)
