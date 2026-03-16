from .detector3d_template import Detector3DTemplate
from ..backbones_3d.cfe import __all__ as ceb_modules
from ..feature_fusion import __all__ as fusion_modules


class PointPillar(Detector3DTemplate):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
        # Extend the default module topology to optionally include radar cluster branch and fusion
        self.module_topology = [
            'vfe', 'backbone_3d', 'map_to_bev_module',
            'ceb', 'fusion_module', 'pfe',
            'backbone_2d', 'dense_head', 'point_head', 'roi_head'
        ]
        self.module_list = self.build_networks()

    def forward(self, batch_dict):
        for cur_module in self.module_list:
            batch_dict = cur_module(batch_dict)

        if self.training:
            loss, tb_dict, disp_dict = self.get_training_loss()

            ret_dict = {
                'loss': loss
            }
            return ret_dict, tb_dict, disp_dict
        else:
            pred_dicts, recall_dicts = self.post_processing(batch_dict)
            return pred_dicts, recall_dicts

    def build_ceb(self, model_info_dict):
        """
        Build the optional Cluster Enhancement Branch that creates radar BEV features.
        """
        if self.model_cfg.get('CEB', None) is None:
            return None, model_info_dict

        ceb_cfg = self.model_cfg.CEB
        if ceb_cfg.NAME not in ceb_modules:
            raise KeyError(f"Unknown CEB module: {ceb_cfg.NAME}")
        ceb_module = ceb_modules[ceb_cfg.NAME](ceb_cfg)
        model_info_dict['module_list'].append(ceb_module)
        return ceb_module, model_info_dict

    def build_fusion_module(self, model_info_dict):
        """
        Build the optional fusion module that fuses PEB and CEB BEV features.
        """
        if self.model_cfg.get('FUSION_MODULE', None) is None:
            return None, model_info_dict

        fusion_cfg = self.model_cfg.FUSION_MODULE
        if fusion_cfg.NAME not in fusion_modules:
            raise KeyError(f"Unknown fusion module: {fusion_cfg.NAME}")
        fusion_module = fusion_modules[fusion_cfg.NAME](fusion_cfg)
        model_info_dict['module_list'].append(fusion_module)
        # After fusion the BEV channel count is defined by the fusion module itself.
        model_info_dict['num_bev_features'] = fusion_cfg.CHANNELS
        return fusion_module, model_info_dict

    def get_training_loss(self):
        disp_dict = {}

        loss_rpn, tb_dict = self.dense_head.get_loss()
        tb_dict = {
            'loss_rpn': loss_rpn.item(),
            **tb_dict
        }

        loss = loss_rpn
        return loss, tb_dict, disp_dict
