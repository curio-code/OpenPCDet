import os
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml
from easydict import EasyDict

from pcdet.datasets.kitti.kitti_dataset import KittiDataset
from pcdet.models.backbones_3d.vfe.pillar_vfe import PillarEnhancementBranch


def _edictify(obj):
    """Recursively convert dicts inside lists to EasyDict for config compatibility."""
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


def _load_model_vfe_cfg(repo_root: Path) -> EasyDict:
    model_cfg_path = repo_root / "tools/cfgs/kitti_models/pointpillar_vod.yaml"
    with open(model_cfg_path, "r") as f:
        cfg = yaml.safe_load(f)
    vfe_cfg = cfg["MODEL"]["VFE"]
    return _edictify(vfe_cfg), cfg.get("CLASS_NAMES", [])


@pytest.mark.parametrize("num_frames", [2])
def test_pillar_enhancement_branch_forward_on_vod(num_frames):
    repo_root = Path(__file__).resolve().parents[1]

    # Load dataset and VFE configs
    dataset_cfg = _load_dataset_cfg(repo_root)
    vfe_cfg, class_names = _load_model_vfe_cfg(repo_root)
    assert class_names, "CLASS_NAMES must be defined in pointpillar_vod.yaml"

    # Determine voxel size from dataset config
    voxel_size = None
    for proc in dataset_cfg.DATA_PROCESSOR:
        if proc.NAME == "transform_points_to_voxels":
            voxel_size = proc.VOXEL_SIZE
            break
    assert voxel_size is not None, "VOXEL_SIZE not found in DATA_PROCESSOR config"

    # Build dataset pointing to VOD 5-frame KITTI-style data
    root_path = Path(dataset_cfg.DATA_PATH)
    assert root_path.exists(), f"Dataset path does not exist: {root_path}"

    ds = KittiDataset(dataset_cfg=dataset_cfg, class_names=class_names, training=False, root_path=root_path)
    assert len(ds) > 0, "Dataset is empty; expected VOD val split with infos present"

    # Build the PillarEnhancementBranch with config
    point_cloud_range = np.array(dataset_cfg.POINT_CLOUD_RANGE, dtype=np.float32)
    num_point_features = ds.point_feature_encoder.num_point_features

    vfe = PillarEnhancementBranch(
        model_cfg=vfe_cfg,
        num_point_features=num_point_features,
        voxel_size=voxel_size,
        point_cloud_range=point_cloud_range,
    )
    vfe.eval()

    # Pull a few frames and run forward
    n = min(num_frames, len(ds))
    for i in range(n):
        data = ds[i]

        # Ensure voxelization happened
        assert "voxels" in data and "voxel_num_points" in data and "voxel_coords" in data

        voxels = torch.from_numpy(data["voxels"]).float()
        voxel_num_points = torch.from_numpy(data["voxel_num_points"]).int()

        # Collate behavior adds batch index as first column to coordinates
        np_coords = data["voxel_coords"]
        if np_coords.shape[1] == 3:
            np_coords = np.pad(np_coords, ((0, 0), (1, 0)), mode="constant", constant_values=0)
        voxel_coords = torch.from_numpy(np_coords).int()

        batch_dict = {
            "voxels": voxels,
            "voxel_num_points": voxel_num_points,
            "voxel_coords": voxel_coords,
        }

        out = vfe(batch_dict)
        assert "pillar_features" in out
        feats = out["pillar_features"]

        # Expected shapes: (num_voxels, last_num_filters)
        assert feats.dim() == 2
        assert feats.shape[0] == voxels.shape[0]
        assert feats.shape[1] == vfe.get_output_feature_dim()

        # Sanity: no NaNs/Infs and not all zeros
        assert torch.isfinite(feats).all()
        assert torch.any(feats != 0)