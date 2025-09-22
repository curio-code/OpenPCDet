# pcdet/models/ce_modules/cluster_enhancement_branch.py
import math
import torch
import torch.nn as nn
from typing import Dict, List, Tuple

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
        super().__init__()
        self.model_cfg = model_cfg
        self.out_channels = self.model_cfg.OUT_CHANNELS #64

        # sensor + clustering params
        self.eps_deg = self.model_cfg.EPS_DEG #, 1.5)        # angular res
        self.range_res = self.model_cfg.RANGE_RES # 0.2)    # Lr
        self.min_pts = self.model_cfg.MIN_PTS # 10)
        self.vel_filter = self.model_cfg.VEL_FILTER # 'mad') # 'mad' or 'fixed'
        self.vel_thresh = self.model_cfg.VEL_THRESH #, 2.0)

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
    def _adaptive_eps(self, xy):
        d = torch.sqrt((xy[:, 0] ** 2) + (xy[:, 1] ** 2))
        Leps = 2.0 * d * math.sin(math.radians(self.eps_deg) * 0.5)
        floor_eps = 2.0 * float(self.range_res)
        Eps = torch.maximum(Leps, torch.as_tensor(floor_eps, device=xy.device, dtype=xy.dtype))
        return Eps

    @staticmethod
    def _xy_to_cell(x: float, y: float, xmin: float, ymin: float, inv_cell: float) -> Tuple[int, int]:
        cx = int(math.floor((x - xmin) * inv_cell))
        cy = int(math.floor((y - ymin) * inv_cell))
        return cx, cy

    @torch.no_grad()
    def _build_grid(self, xy_cpu: torch.Tensor, pc_range: List[float], cell_size: float) -> Dict[Tuple[int, int], List[int]]:
        xmin, ymin = float(pc_range[0]), float(pc_range[1])
        inv_cell = 1.0 / float(cell_size)
        grid: Dict[Tuple[int, int], List[int]] = {}
        x = xy_cpu[:, 0].tolist()
        y = xy_cpu[:, 1].tolist()
        for i in range(len(x)):
            cx, cy = self._xy_to_cell(x[i], y[i], xmin, ymin, inv_cell)
            grid.setdefault((cx, cy), []).append(i)
        return grid

    @torch.no_grad()
    def _region_query(self,
                      idx: int,
                      xy_cpu: torch.Tensor,
                      eps_cpu: torch.Tensor,
                      grid: Dict[Tuple[int, int], List[int]],
                      pc_range: List[float],
                      cell_size: float) -> List[int]:
        xmin, ymin = float(pc_range[0]), float(pc_range[1])
        inv_cell = 1.0 / float(cell_size)
        px = float(xy_cpu[idx, 0])
        py = float(xy_cpu[idx, 1])
        r = float(eps_cpu[idx])

        # compute search window in grid coords
        rc = int(math.ceil(r * inv_cell))
        cx, cy = self._xy_to_cell(px, py, xmin, ymin, inv_cell)

        cand: List[int] = []
        for gy in range(cy - rc, cy + rc + 1):
            for gx in range(cx - rc, cx + rc + 1):
                lst = grid.get((gx, gy))
                if lst:
                    cand.extend(lst)

        if not cand:
            return []

        cand_t = torch.as_tensor(cand, dtype=torch.long)
        diff = xy_cpu[cand_t] - xy_cpu[idx]
        d2 = (diff[:, 0] ** 2) + (diff[:, 1] ** 2)
        mask = d2 <= (r * r + 1e-8)
        return cand_t[mask].tolist()

    @torch.no_grad()
    def _dbscan_variable_eps_single(self, xy: torch.Tensor, eps: torch.Tensor, pc_range: List[float]) -> torch.Tensor:
        N = xy.shape[0]
        if N == 0:
            return torch.empty((0,), dtype=torch.int32, device=xy.device)

        # Work on CPU for neighbor search with Python structures
        xy_cpu = xy.detach().cpu()
        eps_cpu = eps.detach().cpu()
        labels = torch.full((N,), -1, dtype=torch.int32)
        visited = torch.zeros((N,), dtype=torch.bool)

        cell = max(2.0 * float(self.range_res), 1e-3)
        grid = self._build_grid(xy_cpu, pc_range, cell)

        cluster_id = 0
        for i in range(N):
            if visited[i].item():
                continue
            visited[i] = True
            neighbors = self._region_query(i, xy_cpu, eps_cpu, grid, pc_range, cell)
            if len(neighbors) < int(self.min_pts):
                labels[i] = -1
                continue
            # start a new cluster
            labels[i] = cluster_id
            seeds = set(neighbors)
            if i in seeds:
                seeds.remove(i)
            while seeds:
                j = seeds.pop()
                if not visited[j].item():
                    visited[j] = True
                    nhood = self._region_query(j, xy_cpu, eps_cpu, grid, pc_range, cell)
                    if len(nhood) >= int(self.min_pts):
                        seeds.update(nhood)
                if labels[j].item() == -1:
                    labels[j] = cluster_id
            cluster_id += 1

        return labels.to(xy.device)

    @torch.no_grad()
    def _cluster_ids_adaptive(self, xy: torch.Tensor, eps: torch.Tensor, bs_idx: torch.Tensor, pc_range) -> torch.Tensor:
        # Cluster per batch index; ensure unique cluster IDs across batches by offsetting
        if xy.numel() == 0:
            return torch.empty((0,), dtype=torch.int32, device=xy.device)

        uniq_bs = torch.unique(bs_idx).tolist()
        out = torch.empty((xy.shape[0],), dtype=torch.int32, device=xy.device)
        base = 0
        for b in uniq_bs:
            mask = (bs_idx == b)
            idxs = torch.nonzero(mask, as_tuple=False).squeeze(1)
            labels_b = self._dbscan_variable_eps_single(xy[idxs], eps[idxs], pc_range)
            # remap noise stays -1; valid clusters get offset
            pos = labels_b >= 0
            labels_b_out = torch.full_like(labels_b, -1)
            labels_b_out[pos] = labels_b[pos] + base
            out[idxs] = labels_b_out
            max_label = labels_b.max().item() if labels_b.numel() > 0 else -1
            base += (max_label + 1) if max_label >= 0 else 0
        return out

    def _velocity_prune(self, v_abs, cluster_ids):
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
        """
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
        v_src = v_r_comp if torch.any(v_r_comp != 0) or not torch.all(torch.isnan(v_r_comp)) else v_r
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
