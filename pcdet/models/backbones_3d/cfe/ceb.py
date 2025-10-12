# pcdet/models/ce_modules/cluster_enhancement_branch.py
import math
import torch
import torch.nn as nn

class ClusterEnhancementBranch(nn.Module):
    """
    Produces a BEV feature map from radar points via:
      1) adaptive-radius clustering on (x,y) with eps(d)
      2) velocity-consistency pruning
      3) high-res (2H,2W) projection with [count, v_abs] channels
      4) small CBR(+maxpool) -> (C2, H, W)
    Expects batch_dict to carry:
      - batch_dict['points_radar']: [N, 8] = [bs_idx, x,y,z, RCS, v_rel, v_abs, t]
      - batch_dict['spatial_features'] (from PEB) for H,W, pc_range if you want to assert shapes
      - batch_dict['voxel_size'], batch_dict['point_cloud_range'], batch_dict['spatial_features_stride']
    """
    def __init__(self, model_cfg):
        """Store configuration parameters and build the radar encoder components."""
        super().__init__()
        self.model_cfg = model_cfg
        self.out_channels = self.model_cfg.OUT_CHANNELS #64

        # sensor + clustering params
        self.eps_deg = self.model_cfg.EPS_DEG #, 1.5)        # angular res
        self.range_res = self.model_cfg.RANGE_RES # 0.2)    # Lr
        self.min_pts = self.model_cfg.MIN_PTS # 10)
        self.vel_filter = self.model_cfg.VEL_FILTER # 'mad') # 'mad' or 'fixed'
        self.vel_thresh = self.model_cfg.VEL_THRESH #, 2.0)
        
        self.use_vr = self.model_cfg.USE_VR
        self.use_vr_comp = self.model_cfg.USE_VR_COMP

        # optional static pc_range for projection if batch_dict lacks it
        self.pc_range = self.model_cfg.POINT_CLOUD_RANGE

        # encoder from 2×(2H,2W) -> (C2, H, W)
        self.enc = nn.Sequential(
            nn.Conv2d(2, 32, 3, 1, 1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # (2H,2W) -> (H,W)
            nn.Conv2d(32, self.out_channels, 3, 1, 1),
            nn.BatchNorm2d(self.out_channels), nn.ReLU(inplace=True),
        )

    @torch.no_grad()
    #verified
    def _adaptive_eps(self, xy):
        """Compute an adaptive distance threshold per radar point using angular resolution."""
        d = torch.sqrt((xy[:, 0] ** 2) + (xy[:, 1] ** 2))
        Leps = 2.0 * d * math.sin(math.radians(self.eps_deg) * 0.5)
        floor_eps = 2.0 * float(self.range_res)
        Eps = torch.maximum(Leps, torch.as_tensor(floor_eps, device=xy.device, dtype=xy.dtype))
        return Eps

    @torch.no_grad()
    def _dbscan_variable_eps_single(self, xy: torch.Tensor, eps: torch.Tensor) -> torch.Tensor:
        """Run DBSCAN with per-point radii using a vectorized distance matrix."""
        N = xy.shape[0]
        if N == 0:
            return torch.empty((0,), dtype=torch.int32, device=xy.device)

        dist = torch.cdist(xy, xy, p=2)
        neighbor_mask = dist <= eps.unsqueeze(1)
        core_mask = neighbor_mask.sum(dim=1) >= int(self.min_pts)

        labels = torch.full((N,), -1, dtype=torch.int32, device=xy.device)
        unprocessed_core = core_mask.clone()
        cluster_id = 0

        while torch.any(unprocessed_core):
            seed_idx = torch.nonzero(unprocessed_core, as_tuple=False)[0, 0].item()
            cluster_mask = torch.zeros((N,), dtype=torch.bool, device=xy.device)
            frontier = torch.zeros((N,), dtype=torch.bool, device=xy.device)

            cluster_mask[seed_idx] = True
            frontier[seed_idx] = True

            while frontier.any():
                frontier_idx = frontier.nonzero(as_tuple=False).squeeze(1)
                neighbors = neighbor_mask[frontier_idx]  # (F, N) boolean adjacency
                reachable = neighbors.any(dim=0)
                new_points = reachable & (~cluster_mask)

                cluster_mask |= new_points
                frontier = new_points & core_mask

            labels[cluster_mask] = cluster_id
            unprocessed_core &= ~cluster_mask
            cluster_id += 1

        return labels

    @torch.no_grad()
    def _cluster_ids_adaptive(self, xy: torch.Tensor, eps: torch.Tensor, bs_idx: torch.Tensor, pc_range=None) -> torch.Tensor:
        """Cluster each batch independently and remap ids to remain unique across batches."""
        # Cluster per batch index; ensure unique cluster IDs across batches by offsetting
        if xy.numel() == 0:
            return torch.empty((0,), dtype=torch.int32, device=xy.device)

        uniq_bs = torch.unique(bs_idx).tolist()
        out = torch.empty((xy.shape[0],), dtype=torch.int32, device=xy.device)
        base = 0
        for b in uniq_bs:
            mask = (bs_idx == b)
            idxs = torch.nonzero(mask, as_tuple=False).squeeze(1)
            labels_b = self._dbscan_variable_eps_single(xy[idxs], eps[idxs])
            # remap noise stays -1; valid clusters get offset
            pos = labels_b >= 0
            labels_b_out = torch.full_like(labels_b, -1)
            labels_b_out[pos] = labels_b[pos] + base
            out[idxs] = labels_b_out
            max_label = labels_b.max().item() if labels_b.numel() > 0 else -1
            base += (max_label + 1) if max_label >= 0 else 0
        return out

    def _velocity_prune(self, v_abs, cluster_ids):
        """Reject velocity outliers inside each cluster via MAD or a fixed threshold."""
        keep = torch.ones_like(cluster_ids, dtype=torch.bool)
        for cid in cluster_ids.unique():
            if cid.item() < 0: continue
            mask = cluster_ids == cid
            v = v_abs[mask]
            if v.numel() == 0: continue
            v_med = v.median()
            if self.vel_filter == 'mad':
                mad = (v - v_med).abs().median() + 1e-6
                z = (v - v_med).abs() / mad
                keep_idx = z < 3.5
            else:
                keep_idx = (v - v_med).abs() < self.vel_thresh
            keep[mask] = keep_idx
        return keep

    def _bev_project_2x(self, xy, v_abs, bs_idx, pc_range, twoH, twoW):
        """Project retained points into a 2× BEV image with count and mean velocity channels."""
        xmin, ymin, zmin, xmax, ymax, zmax = pc_range
        res_x = (xmax - xmin) / twoW   # note: W maps to x-range in PCDet BEV
        res_y = (ymax - ymin) / twoH   # and H maps to y-range (row-major)

        B = int(bs_idx.max().item()) + 1 if bs_idx.numel() > 0 else 1
        bev = xy.new_zeros((B, 2, twoH, twoW))

        # Convert to BEV indices (row=y, col=x)
        ix = torch.clamp(((xy[:,0]-xmin) / res_x).long(), 0, twoW-1)
        iy = torch.clamp(((xy[:,1]-ymin) / res_y).long(), 0, twoH-1)

        # Count
        bev.index_put_((bs_idx, torch.zeros_like(ix), iy, ix),
                       torch.ones_like(v_abs), accumulate=True)

        # Mean v_abs
        vel_sum = xy.new_zeros((B, twoH, twoW))
        vel_sum.index_put_((bs_idx, iy, ix), v_abs, accumulate=True)
        cnt = bev[:, 0].clamp(min=1e-6)
        bev[:,1] = vel_sum / cnt
        return bev

    def forward(self, batch_dict):
        """Run clustering, pruning, projection, and encoding to produce radar BEV features.
        Writes batch_dict['spatial_features_ceb'] with shape [B, C2, H, W]
        """
        # Acquire radar points: prefer 'points' (standard in PCDet) then optional 'points_radar'
        pts = batch_dict.get('points', None)
        if pts is None:
            pts = batch_dict.get('points_radar', None)
        assert pts is not None and pts.shape[1] >= 8, (
            "Expected 'points' with columns [bs,x,y,z,rcs,v_r,v_r_comp,time] or 'points_radar'"
        )

        bs_idx = pts[:, 0].long()
        xy = pts[:, 1:3]
        # choose velocity source: prefer compensated if present
        # columns: [bs, x, y, z, rcs, v_r, v_r_comp, time]
        v_r_comp = pts[:, 6]
        v_r = pts[:, 5]
        v_src = v_r_comp if self.use_vr_comp or not self.use_vr else v_r
        v_abs = v_src.abs()

        # derive H,W from existing BEV (from PEB scatter)
        spatial_features = batch_dict['spatial_features']   # [B, Cpeb, H, W]
        B, _, H, W = spatial_features.shape
        twoH, twoW = 2*H, 2*W

        # clustering + prune
        eps = self._adaptive_eps(xy)
        # prefer pc_range from batch_dict; fallback to module's static pc_range if provided
        pc_range = batch_dict.get('point_cloud_range', self.pc_range)
        assert pc_range is not None, "point_cloud_range must be provided in batch_dict or model_cfg"
        if isinstance(pc_range, torch.Tensor):
            pc_range_list = pc_range.detach().cpu().tolist()
        elif isinstance(pc_range, (list, tuple)):
            pc_range_list = list(pc_range)
        else:
            pc_range_list = list(pc_range)

        cluster_ids = self._cluster_ids_adaptive(xy, eps, bs_idx, pc_range_list)
        keep = self._velocity_prune(v_abs, cluster_ids)

        xy = xy[keep]; v_abs = v_abs[keep]; bs_idx = bs_idx[keep]

        bev2x = self._bev_project_2x(
            xy, v_abs, bs_idx,
            pc_range_list, twoH, twoW
        )  # [B,2,2H,2W]

        feat = self.enc(bev2x)  # [B,C2,H,W]
        batch_dict['spatial_features_ceb'] = feat
        return batch_dict
