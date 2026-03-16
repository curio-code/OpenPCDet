import torch
import torch.nn as nn
import torch.nn.functional as F

from .vfe_template import VFETemplate


class CapsVFE(VFETemplate):
    """
    Content-Adaptive Pillar Segmentation (CAPS) VFE.

    Dual-path architecture for sparse 4D radar point clouds:
      1. Point-level path: per-point features -> Linear -> BN -> ReLU -> max-pool
      2. Histogram path: z-binned statistics -> BN -> depthwise-sep conv1d -> segment pooling
      3. Fusion: concatenate both paths -> Linear -> output

    The point-level path preserves fine-grained per-point information (critical
    for extremely sparse radar with 1-3 points per pillar), while the histogram
    path captures vertical structure patterns across the z-axis.
    """

    def __init__(self, model_cfg, num_point_features, voxel_size, point_cloud_range, **kwargs):
        super().__init__(model_cfg=model_cfg)

        self.num_height_bins = self.model_cfg.get('NUM_HEIGHT_BINS', 4)
        self.num_segments = self.model_cfg.get('NUM_SEGMENTS', 2)
        self.mid_channels = self.model_cfg.get('MID_CHANNELS', 32)
        self.num_output_features = self.model_cfg.get('NUM_OUTPUT_FEATURES', 64)
        self.use_segments = self.model_cfg.get('USE_SEGMENTS', True)

        # Feature indices in raw voxel tensor: [x, y, z, rcs, v_r, v_r_comp, time]
        self.rcs_idx = self.model_cfg.get('RCS_FEATURE_IDX', 3)
        self.vel_idx = self.model_cfg.get('VELOCITY_FEATURE_IDX', 5)
        self.time_idx = self.model_cfg.get('TIME_FEATURE_IDX', 6)

        # Z-range for histogram binning
        self.z_min = point_cloud_range[2]
        self.z_max = point_cloud_range[5]

        # Voxel geometry for pillar-center offsets
        self.voxel_x = voxel_size[0]
        self.voxel_y = voxel_size[1]
        self.voxel_z = voxel_size[2]
        self.x_offset = self.voxel_x / 2 + point_cloud_range[0]
        self.y_offset = self.voxel_y / 2 + point_cloud_range[1]
        self.z_offset = self.voxel_z / 2 + point_cloud_range[2]

        K = self.num_segments

        # ---- Point-level branch ----
        # Per-point: xyz(3) + rcs(1) + vel(1) + time(1) + cluster_off(3) + center_off(3) + vel_diff(1) = 13
        point_in_dim = 13
        self.point_linear = nn.Linear(point_in_dim, self.mid_channels, bias=False)
        self.point_bn = nn.BatchNorm1d(self.mid_channels, eps=1e-3, momentum=0.01)

        # ---- Histogram branch ----
        # Per-bin: occupancy(1) + norm_count(1) + mean_rcs(1) + mean_vel(1) + mean_z_norm(1) = 5
        hist_channels = 5
        self.hist_input_bn = nn.BatchNorm1d(hist_channels)
        self.dw_conv = nn.Conv1d(
            hist_channels, hist_channels, kernel_size=3, padding=1, groups=hist_channels
        )
        self.dw_bn = nn.BatchNorm1d(hist_channels)
        self.pw_conv = nn.Conv1d(hist_channels, self.mid_channels, kernel_size=1)
        self.pw_bn = nn.BatchNorm1d(self.mid_channels)

        if self.use_segments:
            self.segment_head = nn.Linear(self.mid_channels, K)
            self.segment_scale = nn.Embedding(K, self.mid_channels)
            hist_out_dim = K * self.mid_channels
        else:
            hist_out_dim = self.mid_channels

        # ---- Fusion projection ----
        self.output_proj = nn.Linear(self.mid_channels + hist_out_dim, self.num_output_features)

    def get_output_feature_dim(self):
        return self.num_output_features

    def get_paddings_indicator(self, actual_num, max_num, axis=0):
        actual_num = torch.unsqueeze(actual_num, axis + 1)
        max_num_shape = [1] * len(actual_num.shape)
        max_num_shape[axis + 1] = -1
        max_num = torch.arange(max_num, dtype=torch.int, device=actual_num.device).view(max_num_shape)
        paddings_indicator = actual_num.int() > max_num
        return paddings_indicator

    def forward(self, batch_dict, **kwargs):
        voxel_features = batch_dict['voxels']               # (N, T, C)
        voxel_num_points = batch_dict['voxel_num_points']    # (N,)
        coords = batch_dict['voxel_coords']                  # (N, 4) [batch, z, y, x]

        N, T, C = voxel_features.shape
        B = self.num_height_bins
        device = voxel_features.device
        dtype = voxel_features.dtype
        eps = 1e-6

        num_pts = voxel_num_points.to(dtype).clamp(min=1.0)  # (N,)

        # ---- Padding mask ----
        mask = self.get_paddings_indicator(voxel_num_points, T, axis=0)  # (N, T) bool
        mask_3d = mask.unsqueeze(-1).to(dtype)                            # (N, T, 1)

        # ---- Shared geometric features ----
        xyz = voxel_features[:, :, :3]  # (N, T, 3)

        # Cluster-mean offset (offset from mean of real points in pillar)
        points_mean = (xyz * mask_3d).sum(dim=1, keepdim=True) / num_pts.view(-1, 1, 1)  # (N, 1, 3)
        f_cluster = xyz - points_mean  # (N, T, 3)

        # Pillar-center offset
        f_center = torch.zeros_like(xyz)  # (N, T, 3)
        f_center[:, :, 0] = voxel_features[:, :, 0] - (
            coords[:, 3].to(dtype).unsqueeze(1) * self.voxel_x + self.x_offset)
        f_center[:, :, 1] = voxel_features[:, :, 1] - (
            coords[:, 2].to(dtype).unsqueeze(1) * self.voxel_y + self.y_offset)
        f_center[:, :, 2] = voxel_features[:, :, 2] - (
            coords[:, 1].to(dtype).unsqueeze(1) * self.voxel_z + self.z_offset)

        # Velocity difference (per-point vel minus pillar-mean vel)
        vel = voxel_features[:, :, self.vel_idx:self.vel_idx + 1]  # (N, T, 1)
        vel_mean = (vel * mask_3d).sum(dim=1, keepdim=True) / num_pts.view(-1, 1, 1)  # (N, 1, 1)
        vel_diff = vel - vel_mean  # (N, T, 1)

        # Raw scalar features
        rcs = voxel_features[:, :, self.rcs_idx:self.rcs_idx + 1]        # (N, T, 1)
        time_feat = voxel_features[:, :, self.time_idx:self.time_idx + 1]  # (N, T, 1)

        # ==================================================
        # POINT-LEVEL BRANCH
        # ==================================================
        # xyz(3) + rcs(1) + vel(1) + time(1) + cluster(3) + center(3) + vel_diff(1) = 13
        point_feats = torch.cat([
            xyz, rcs, vel, time_feat, f_cluster, f_center, vel_diff
        ], dim=-1)  # (N, T, 13)
        point_feats = point_feats * mask_3d  # (N, T, 13)

        pf = self.point_linear(point_feats)  # (N, T, mid_channels)
        # BN on (N, C, T) format — disable cudnn to match PFNLayer behavior
        pf_flat = pf.view(-1, self.mid_channels)  # (N*T, mid_channels)
        pf_flat = self.point_bn(pf_flat)
        pf = pf_flat.view(N, T, self.mid_channels)  # (N, T, mid_channels)
        pf = F.relu(pf)
        pf = pf * mask_3d  # re-mask after BN (BN shifts padded positions away from zero)
        point_out = pf.max(dim=1)[0]  # (N, mid_channels)

        # ==================================================
        # HISTOGRAM BRANCH
        # ==================================================
        z_vals = voxel_features[:, :, 2]  # (N, T)
        z_range = self.z_max - self.z_min
        bin_idx = ((z_vals - self.z_min) / (z_range + eps) * B).long().clamp(0, B - 1)  # (N, T)
        bin_idx = bin_idx * mask.long()  # padded points -> bin 0 (zeroed by mask below)

        bin_idx_1 = bin_idx.unsqueeze(-1)  # (N, T, 1)
        mask_1d = mask.to(dtype)           # (N, T)

        # Count per bin
        count = torch.zeros(N, B, 1, device=device, dtype=dtype)
        count.scatter_add_(1, bin_idx_1, mask_1d.unsqueeze(-1))  # (N, B, 1)

        # Occupancy: binary indicator
        occupancy = (count > 0).to(dtype)  # (N, B, 1)

        # Normalized count: fraction of pillar's points in each bin
        norm_count = count / (num_pts.view(-1, 1, 1) + eps)  # (N, B, 1)

        # Mean RCS per bin
        rcs_sum = torch.zeros(N, B, 1, device=device, dtype=dtype)
        rcs_sum.scatter_add_(1, bin_idx_1, (voxel_features[:, :, self.rcs_idx] * mask_1d).unsqueeze(-1))
        rcs_mean = rcs_sum / (count + eps)  # (N, B, 1)

        # Mean velocity per bin
        vel_sum = torch.zeros(N, B, 1, device=device, dtype=dtype)
        vel_sum.scatter_add_(1, bin_idx_1, (voxel_features[:, :, self.vel_idx] * mask_1d).unsqueeze(-1))
        vel_mean_bin = vel_sum / (count + eps)  # (N, B, 1)

        # Mean normalized z per bin
        z_norm = (z_vals - self.z_min) / (z_range + eps) * mask_1d  # (N, T) in [0, 1]
        z_sum = torch.zeros(N, B, 1, device=device, dtype=dtype)
        z_sum.scatter_add_(1, bin_idx_1, z_norm.unsqueeze(-1))
        z_mean = z_sum / (count + eps)  # (N, B, 1)

        histogram = torch.cat([occupancy, norm_count, rcs_mean, vel_mean_bin, z_mean], dim=-1)  # (N, B, 5)

        # BN -> DW-conv -> BN -> ReLU -> PW-conv -> BN -> ReLU
        h = histogram.permute(0, 2, 1)                   # (N, 5, B)
        h = self.hist_input_bn(h)
        h = F.relu(self.dw_bn(self.dw_conv(h)))           # (N, 5, B)
        h = F.relu(self.pw_bn(self.pw_conv(h)))            # (N, mid_channels, B)
        h = h.permute(0, 2, 1)                            # (N, B, mid_channels)

        if self.use_segments:
            K = self.num_segments
            scores = torch.sigmoid(self.segment_head(h))  # (N, B, K)

            segments = []
            for k in range(K):
                w = scores[:, :, k:k + 1]                        # (N, B, 1)
                weighted_sum = (w * h).sum(dim=1)                 # (N, mid_channels)
                weight_sum = w.sum(dim=1) + eps                   # (N, 1)
                segments.append(weighted_sum / weight_sum)        # (N, mid_channels)

            scale_idx = torch.arange(K, device=device)
            scales = torch.sigmoid(self.segment_scale(scale_idx))  # (K, mid_channels)
            for k in range(K):
                segments[k] = segments[k] * scales[k]

            hist_out = torch.cat(segments, dim=-1)  # (N, K * mid_channels)
        else:
            hist_out = h.mean(dim=1)  # (N, mid_channels)

        # ==================================================
        # FUSION
        # ==================================================
        fused = torch.cat([point_out, hist_out], dim=-1)   # (N, mid_channels + hist_out_dim)
        pillar_features = self.output_proj(fused)           # (N, num_output_features)

        batch_dict['pillar_features'] = pillar_features
        return batch_dict
