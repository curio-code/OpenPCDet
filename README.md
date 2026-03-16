# 4DRadDet Implementation Notes

This document summarizes the OpenPCDet integration of the 4DRadDet architecture, _4D Radar Object Detection for Autonomous Driving_, and how the radar-centric modules are organized inside the codebase. Use it alongside the original paper when configuring experiments on the View-of-Delft (VoD) radar benchmark or similar datasets.

## High-level pipeline

The implementation mirrors the paper's three-branch design:

1. **Pillar Enhancement Branch (PEB)** extracts per-pillar radar features with configurable inputs such as RCS, compensated velocity, timestamps, and an optional velocity-difference cue before passing them through stacked PFN layers.【F:pcdet/models/backbones_3d/vfe/pillar_vfe.py†L273-L460】
2. **Cluster Enhancement Branch (CEB)** groups radar points with an adaptive-radius DBSCAN, prunes clusters via velocity consistency, projects the survivors into a 2× BEV image, and encodes them into an H×W map.【F:pcdet/models/backbones_3d/cfe/ceb.py†L6-L207】
3. **Cross-Attention Feature Fusion (CAFF)** performs multi-head cross-attention between PEB (keys/values) and CEB (queries), applies a feed-forward refinement with learnable residual scaling, and fuses the results with a squeeze-and-excitation block plus a 1×1 convolution.【F:pcdet/models/feature_fusion/caff.py†L44-L159】

The modules are registered so they can be referenced directly from YAML configs under `MODEL.VFE`, `MODEL.BACKBONE_3D`, or fusion blocks.【F:pcdet/models/backbones_3d/vfe/__init__.py†L1-L20】【F:pcdet/models/backbones_3d/cfe/__init__.py†L1-L4】【F:pcdet/models/feature_fusion/__init__.py†L1-L4】

## Data prerequisites

* The VoD radar conversion follows the KITTI reader, so VoD samples must be exported into a KITTI-format folder with `kitti_infos_*.pkl` metadata. The helper in `kitti_dataset.py` points the info generator to `data/view_of_delft/radar_5frames` for convenience.【F:pcdet/datasets/kitti/kitti_dataset.py†L485-L498】
* Unit tests rely on a dataset config at `tools/cfgs/dataset_configs/radar_5frames_as_kitti_dataset.yaml` and a PointPillars-derived model config named `pointpillar_vod.yaml`, both of which should define the voxel grid and radar feature usage expected by the modules.【F:tests/test_pillar_enhancement_branch.py†L23-L71】【F:tests/test_cluster_enhancement_branch.py†L24-L141】

Ensure the dataset cfg exposes the standard KITTI processors (especially `transform_points_to_voxels`) so the PFN layers receive consistent voxel sizes.【F:tests/test_pillar_enhancement_branch.py†L47-L54】

## Pillar Enhancement Branch (PEB)

* Validates that radar-specific feature toggles (RCS, velocities, timestamps, elevation, velocity difference source) are declared in the YAML config, raising early if anything is missing.【F:pcdet/models/backbones_3d/vfe/pillar_vfe.py†L283-L325】
* Selects which raw radar channels to keep, optionally adds a per-pillar compensated-velocity residual (`v_c`), and appends geometric cues (cluster offsets, center offsets, distance).【F:pcdet/models/backbones_3d/vfe/pillar_vfe.py†L327-L449】
* Applies padding masks before passing tokens through stacked PFN layers and exposes the final tensor through `batch_dict['pillar_features']`.【F:pcdet/models/backbones_3d/vfe/pillar_vfe.py†L448-L460】

The regression test iterates over real VoD frames to confirm voxelization, tensor shapes, and numerical stability, which is useful when modifying feature selections.【F:tests/test_pillar_enhancement_branch.py†L38-L108】

## Cluster Enhancement Branch (CEB)

* Computes a per-point radius from range and angular resolution, then runs a custom DBSCAN variant that preserves unique cluster IDs per batch sample.【F:pcdet/models/backbones_3d/cfe/ceb.py†L45-L115】
* Filters clusters using median absolute deviation or a fixed velocity threshold, projects the survivors into `[count, mean |v|]` channels on a doubled-resolution BEV, and downsamples them with a lightweight CNN encoder.【F:pcdet/models/backbones_3d/cfe/ceb.py†L116-L207】
* Writes the encoded tensor to `batch_dict['spatial_features_ceb']`, ready to be fused with the lidar/radar pillar features.【F:pcdet/models/backbones_3d/cfe/ceb.py†L158-L207】

The accompanying test suite covers both VoD frames and synthetic clusters, and can optionally dump BEV visualizations for debugging the clustering/velocity pruning stages.【F:tests/test_cluster_enhancement_branch.py†L114-L347】

## Cross-Attention Feature Fusion (CAFF)

* Reshapes both BEV feature maps into sequences, augments them with learnable positional encodings, and runs explicit multi-head attention where CEB queries the PEB memory.【F:pcdet/models/feature_fusion/caff.py†L94-L138】
* Introduces four learnable residual scalars (`δ₁…δ₄`) to balance skip connections and the feed-forward block before reprojecting to BEV space.【F:pcdet/models/feature_fusion/caff.py†L71-L152】
* Optionally applies an SE block, concatenates the original and refined pillar tensors, and compresses them with a shared 1×1 convolution to produce the fused radar-lidar representation.【F:pcdet/models/feature_fusion/caff.py†L81-L159】

You can disable SENet or adjust the attention heads/dropout via the module constructor when wiring CAFF into a custom model definition.

## Practical tips

* When extending configs, keep the BEV grid size consistent across branches so CAFF operates on aligned tensors; both PEB and CEB assume the same `(H, W)` derived from the voxelized radar pillars.【F:pcdet/models/backbones_3d/cfe/ceb.py†L179-L207】【F:pcdet/models/backbones_3d/vfe/pillar_vfe.py†L398-L460】
* The provided tests double as smoke tests for data paths—execute them after updating dataset locations or feature toggles to catch misconfigurations early.【F:tests/test_pillar_enhancement_branch.py†L38-L108】【F:tests/test_cluster_enhancement_branch.py†L114-L347】