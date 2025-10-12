import os
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml
from easydict import EasyDict
from skimage import io as skio
from skimage import draw as skdraw

from pcdet.datasets.kitti.kitti_dataset import KittiDataset
from pcdet.models.backbones_3d.cfe import ClusterEnhancementBranch


def _edictify(obj):
    if isinstance(obj, dict):
        return EasyDict({k: _edictify(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [_edictify(v) for v in obj]
    return obj


def _load_dataset_cfg(repo_root: Path) -> EasyDict:
    ds_cfg_path = repo_root / "tools/cfgs/dataset_configs/radar_5frames_as_kitti_dataset.yaml"
    with open(ds_cfg_path, "r") as f:
        cfg = yaml.safe_load(f)
    return _edictify(cfg)


def _save_cluster_image(
    xy,
    cluster_ids,
    keep_mask,
    pc_range,
    out_path: Path,
    H: int,
    W: int,
    gt_boxes=None,
):
    """Render clusters (kept vs pruned) into a 2x BEV image and save."""
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(xy, torch.Tensor):
        xy_np = xy.detach().cpu().numpy()
    else:
        xy_np = np.asarray(xy)
    if isinstance(cluster_ids, torch.Tensor):
        cid_np = cluster_ids.detach().cpu().numpy()
    else:
        cid_np = np.asarray(cluster_ids)
    if isinstance(keep_mask, torch.Tensor):
        keep_np = keep_mask.detach().cpu().numpy().astype(bool)
    else:
        keep_np = np.asarray(keep_mask).astype(bool)

    twoH, twoW = 2 * H, 2 * W
    xmin, ymin, _, xmax, ymax, _ = pc_range
    res_x = (xmax - xmin) / twoW
    res_y = (ymax - ymin) / twoH

    img = np.full((twoH, twoW, 3), 255, dtype=np.uint8)

    def cid_to_color(c: int) -> tuple:
        if c < 0:
            return (255, 255, 255)  # noise
        return (int((37 * c) % 256), int((17 * c) % 256), int((91 * c) % 256))

    def to_pix(coords):
        coords = np.asarray(coords)
        ix = np.clip(((coords[:, 0] - xmin) / res_x).astype(np.int32), 0, twoW - 1)
        iy = np.clip(((coords[:, 1] - ymin) / res_y).astype(np.int32), 0, twoH - 1)
        return iy, ix

    if np.any(~keep_np):
        drop_xy = xy_np[~keep_np]
        dy, dx = to_pix(drop_xy)
        for py, px in zip(dy.tolist(), dx.tolist()):
            img[py, px, :] = (120, 120, 120)

    keep_xy = xy_np[keep_np]
    keep_cids = cid_np[keep_np]
    ky, kx = to_pix(keep_xy)
    for py, px, cid in zip(ky.tolist(), kx.tolist(), keep_cids.tolist()):
        img[py, px, :] = cid_to_color(int(cid))

    if gt_boxes is not None and len(gt_boxes) > 0:
        palette = {
            1: (255, 0, 0),
            2: (0, 255, 0),
            3: (0, 0, 255),
        }
        for b in gt_boxes:
            x, y, z, dx, dy, dz, heading, cls = b
            dx2, dy2 = dx * 0.5, dy * 0.5
            c, s = np.cos(heading), np.sin(heading)
            local = np.array([
                [ dx2,  dy2],
                [ dx2, -dy2],
                [-dx2, -dy2],
                [-dx2,  dy2],
            ], dtype=np.float32)
            R = np.array([[c, -s], [s, c]], dtype=np.float32)
            world_xy = (local @ R.T) + np.array([x, y], dtype=np.float32)
            px = np.clip(((world_xy[:, 0] - xmin) / res_x).astype(np.int32), 0, twoW - 1)
            py = np.clip(((world_xy[:, 1] - ymin) / res_y).astype(np.int32), 0, twoH - 1)
            rr, cc = skdraw.polygon_perimeter(py, px, shape=(twoH, twoW), clip=True)
            color = palette.get(int(cls), (255, 255, 0))
            img[rr, cc, :] = color

    skio.imsave(str(out_path), img)


@pytest.mark.parametrize("num_frames", [1])
def test_cluster_enhancement_branch_forward_on_vod(num_frames):
    repo_root = Path(__file__).resolve().parents[1]

    # Load dataset config and dataset
    dataset_cfg = _load_dataset_cfg(repo_root)
    root_path = Path(dataset_cfg.DATA_PATH)
    assert root_path.exists(), f"Dataset path does not exist: {root_path}"

    # Build dataset and check it has data
    class_names = ['Car', 'Pedestrian', 'Cyclist']
    ds = KittiDataset(dataset_cfg=dataset_cfg, class_names=class_names, training=False, root_path=root_path)
    assert len(ds) > 0, "Dataset is empty; expected VOD val split with infos present"

    # Setup CEB config with defaults
    ceb_cfg = EasyDict(
        dict(
            OUT_CHANNELS=32,
            EPS_DEG=1.5,
            RANGE_RES=0.2,
            MIN_PTS=10,
            VEL_FILTER='mad',
            VEL_THRESH=2.0,
            POINT_CLOUD_RANGE=dataset_cfg.POINT_CLOUD_RANGE,
            USE_VR= False,
            USE_VR_COMP= True,
        )
    )
    ceb = ClusterEnhancementBranch(ceb_cfg)
    ceb.eval()

    # Determine BEV H, W from dataset grid_size (ny, nx)
    nx, ny, nz = ds.grid_size
    assert nz == 1
    H, W = int(ny), int(nx)

    # Pull frames and run CEB forward
    n = min(num_frames, len(ds))
    for i in range(n):
        data = ds[i]

        # points: (N, 7) -> add batch index as first column to form (N, 8)
        assert "points" in data
        pts = data["points"]
        assert pts.shape[1] >= 7
        bs_col = np.zeros((pts.shape[0], 1), dtype=pts.dtype)
        pts_b = np.concatenate([bs_col, pts[:, :7]], axis=1)

        # Build minimal batch_dict expected by CEB
        batch_dict = {
            "points": torch.from_numpy(pts_b).float(),
            "spatial_features": torch.zeros((1, 64, H, W), dtype=torch.float32),
            "point_cloud_range": torch.tensor(dataset_cfg.POINT_CLOUD_RANGE, dtype=torch.float32),
        }

        out = ceb(batch_dict)
        assert "spatial_features_ceb" in out
        feats = out["spatial_features_ceb"]

        # Expected shapes: (1, OUT_CHANNELS, H, W)
        assert feats.dim() == 4
        assert feats.shape[0] == 1
        assert feats.shape[1] == ceb_cfg.OUT_CHANNELS
        assert feats.shape[2] == H and feats.shape[3] == W

        # Sanity: finite and not all zeros
        assert torch.isfinite(feats).all()
        assert torch.any(feats != 0)

        # Additionally: visualize clusters in a 2x BEV image for debugging
        pts_t = torch.from_numpy(pts_b).float()
        bs_idx = pts_t[:, 0].long()
        xy = pts_t[:, 1:3]
        v_r_comp = pts_t[:, 6]
        v_r = pts_t[:, 5]
        v_src = v_r_comp if torch.any(v_r_comp != 0) or not torch.all(torch.isnan(v_r_comp)) else v_r
        v_abs = v_src.abs()

        eps = ceb._adaptive_eps(xy)
        pc_range = dataset_cfg.POINT_CLOUD_RANGE
        cluster_ids = ceb._cluster_ids_adaptive(xy, eps, bs_idx, pc_range)
        keep = ceb._velocity_prune(v_abs, cluster_ids)

        xy_k = xy[keep]
        cid_k = cluster_ids[keep]
        bs_k = bs_idx[keep]

        twoH, twoW = 2 * H, 2 * W
        xmin, ymin, _, xmax, ymax, _ = pc_range
        res_x = (xmax - xmin) / twoW
        res_y = (ymax - ymin) / twoH
        ix = torch.clamp(((xy_k[:, 0] - xmin) / res_x).long(), 0, twoW - 1)
        iy = torch.clamp(((xy_k[:, 1] - ymin) / res_y).long(), 0, twoH - 1)

        assert bs_k.unique().numel() == 1
        img = np.full((twoH, twoW, 3), 255, dtype=np.uint8)

        def cid_to_color(c: int) -> tuple:
            if c < 0:
                return (255, 255, 255)  # noise
            return (int((37 * c) % 256), int((17 * c) % 256), int((91 * c) % 256))

        for py, px, c in zip(iy.tolist(), ix.tolist(), cid_k.tolist()):
            img[py, px, :] = cid_to_color(int(c))

        out_dir = repo_root / "output" / "ceb_debug"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"clusters_frame_{i:06d}.png"
        # Overlay GT boxes if available
        if "gt_boxes" in data and data["gt_boxes"].size > 0:
            gt = data["gt_boxes"]  # (M, 8): x,y,z,dx,dy,dz,heading,label
            # class-based colors (1..3): Car, Pedestrian, Cyclist
            palette = {
                1: (255, 0, 0),    # Car -> red
                2: (0, 255, 0),    # Pedestrian -> green
                3: (0, 0, 255),    # Cyclist -> blue
            }
            for b in gt:
                x, y, z, dx, dy, dz, heading, cls = b
                dx2, dy2 = dx * 0.5, dy * 0.5
                c, s = np.cos(heading), np.sin(heading)
                # local corners (x forward, y left): (dx/2, dy/2) etc.
                local = np.array([
                    [ dx2,  dy2],
                    [ dx2, -dy2],
                    [-dx2, -dy2],
                    [-dx2,  dy2],
                ], dtype=np.float32)
                R = np.array([[c, -s], [s, c]], dtype=np.float32)
                world_xy = (local @ R.T) + np.array([x, y], dtype=np.float32)
                # map to pixel indices
                px = np.clip(((world_xy[:, 0] - xmin) / res_x).astype(np.int32), 0, twoW - 1)
                py = np.clip(((world_xy[:, 1] - ymin) / res_y).astype(np.int32), 0, twoH - 1)
                rr, cc = skdraw.polygon_perimeter(py, px, shape=(twoH, twoW), clip=True)
                color = palette.get(int(cls), (255, 255, 0))
                img[rr, cc, :] = color

        skio.imsave(str(out_path), img)

@torch.no_grad()
def test_ceb_two_random_clusters():
    """Synthetic sanity check: two random clusters with velocity outliers."""
    pc_range = [0.0, -25.6, -3.0, 51.2, 25.6, 2.0]
    ceb_cfg = EasyDict(dict(
        OUT_CHANNELS=16,
        EPS_DEG=1.5,
        RANGE_RES=0.2,
        MIN_PTS=10,
        VEL_FILTER='mad',
        VEL_THRESH=2.5,
        POINT_CLOUD_RANGE=pc_range,
        USE_VR= False,
        USE_VR_COMP= True,
    ))
    ceb = ClusterEnhancementBranch(ceb_cfg)
    ceb.eval()

    rng = np.random.default_rng(42)

    def make_cluster(center, sigma, n, v_mean, v_outlier=None, n_out=0):
        xy = rng.normal(loc=center, scale=(sigma, sigma), size=(n, 2)).astype(np.float32)
        v = (v_mean + 0.1 * rng.standard_normal(n)).astype(np.float32)
        if n_out > 0 and v_outlier is not None:
            v[:n_out] = v_outlier
        return xy, v

    xy1, v1 = make_cluster(center=(10.0, 0.0), sigma=0.2, n=60, v_mean=2.0, v_outlier=6.0, n_out=0)
    xy2, v2 = make_cluster(center=(30.0, 5.0), sigma=0.3, n=60, v_mean=-1.5, v_outlier=-6.0, n_out=0)
    xy3, v3 = make_cluster(center=(15.0, 5.0), sigma=0.3, n=60, v_mean=-1.5, v_outlier=-6.0, n_out=0)

    def pack(xy, v):
        N = xy.shape[0]
        bs = np.zeros((N, 1), dtype=np.float32)
        z = np.zeros((N, 1), dtype=np.float32)
        rcs = np.zeros((N, 1), dtype=np.float32)
        v_r = v.reshape(-1, 1).astype(np.float32)
        v_rc = v.reshape(-1, 1).astype(np.float32)
        t = np.zeros((N, 1), dtype=np.float32)
        return np.concatenate([bs, xy.astype(np.float32), z, rcs, v_r, v_rc, t], axis=1)

    pts_np = np.concatenate([pack(xy1, v1), pack(xy2, v2), pack(xy3, v3)], axis=0)
    pts = torch.from_numpy(pts_np).float()

    voxel = 0.16
    H = int(round((pc_range[4] - pc_range[1]) / voxel))
    W = int(round((pc_range[3] - pc_range[0]) / voxel))
    batch_dict = {
        'points': pts,
        'spatial_features': torch.zeros((1, 64, H, W), dtype=torch.float32),
        'point_cloud_range': torch.tensor(pc_range, dtype=torch.float32),
    }

    out = ceb(batch_dict)
    bev = out['spatial_features_ceb']
    assert bev.shape == (1, ceb_cfg.OUT_CHANNELS, H, W)
    assert torch.isfinite(bev).all() and torch.any(bev != 0)

    bs_idx = pts[:, 0].long()
    xy = pts[:, 1:3]
    v_abs = pts[:, 6].abs()
    eps = ceb._adaptive_eps(xy)
    cluster_ids = ceb._cluster_ids_adaptive(xy, eps, bs_idx, pc_range)
    keep = ceb._velocity_prune(v_abs, cluster_ids)

    kept_xy = xy[keep]
    kept_cids = cluster_ids[keep]
    uniq = torch.unique(kept_cids[kept_cids >= 0]).tolist()
    assert len(uniq) >= 2

    sizes, cents = [], []
    for cid in uniq:
        mask = (kept_cids == cid)
        sizes.append(int(mask.sum().item()))
        cents.append(kept_xy[mask].mean(dim=0).cpu().numpy())
    top_idx = np.argsort(sizes)[-3:]
    centroids = sorted([cents[top_idx[0]], cents[top_idx[1]], cents[top_idx[2]]], key=lambda a: a[0])
    assert np.linalg.norm(centroids[0] - np.array([10.0, 0.0])) < 1.0
    #assert np.linalg.norm(centroids[1] - np.array([30.0, 5.0])) < 1.0

    repo_root = Path(__file__).resolve().parents[1]
    out_dir = repo_root / 'output' / 'ceb_debug'
    out_path = out_dir / 'clusters_two_random.png'
    _save_cluster_image(
        xy,
        cluster_ids,
        keep,
        pc_range,
        out_path,
        H,
        W,
    )

test_cluster_enhancement_branch_forward_on_vod(10)
test_ceb_two_random_clusters()
