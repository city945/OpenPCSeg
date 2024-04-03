import os
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../'))

import unittest
import pu4c
from pprint import pprint

from pcseg.config import cfgs, cfg_from_yaml_file
from pcseg.data import build_dataloader
from pcseg.model import build_network, load_data_to_gpu
import torch

class TestCylinder3D(unittest.TestCase):

    def __init__(self, methodName: str = "runTest") -> None:
        super().__init__(methodName)

        pu4c.nprandom.seed(123)

    # @unittest.skip("")
    def test_semantic_kitti_cylinder3d(self):
        cfg_file = os.path.join('tools', 'cfgs', 'voxel', 'semantic_kitti', 'cylinder_cy480_cr10.yaml')
        ckpt = cfgs.ROOT_DIR / "model_zoo/download/semkitti_cylinder_cy480_cr10_checkpoint_epoch_35.pth"
        cfg_from_yaml_file(cfg_file, cfgs)
        cfgs.DATA.DEBUG = True
        cfgs.MODEL.IF_DIST = False
        batch_size = 2
        rand_idx = pu4c.nprandom.randint(1000)

        train_set, train_loader, train_sampler = build_dataloader(
            data_cfgs=cfgs.DATA,
            modality=cfgs.MODALITY,
            batch_size=batch_size,
            root_path=cfgs.DATA.DATA_PATH,
            workers=0,
            training=True,
        )
        batch_dict = next(iter(train_loader))
        metric_dict = {
            "point_coord": batch_dict["point_coord"][rand_idx],
            "point_feature": batch_dict["point_feature"][rand_idx][:-1],
            "voxel_coord": batch_dict["voxel_coord"][rand_idx],
        }
        pprint(metric_dict)

        

        if cfgs.DATA.DATASET == 'nuscenes':
            num_class = 17
        elif cfgs.DATA.DATASET == 'semantickitti':
            num_class = 20
        elif cfgs.DATA.DATASET == 'waymo':
            num_class = 23
        
        model = build_network(model_cfgs=cfgs.MODEL, num_class=num_class)
        if os.path.exists(ckpt):
            model.load_params_from_file(ckpt, to_cpu=True)
        model.cuda().eval()

        load_data_to_gpu(batch_dict)
        with torch.no_grad():
            ret_dict = model(batch_dict)
        point_predict = ret_dict['point_predict']
        point_labels = ret_dict['point_labels']
        metric_dict = {'acc': (point_predict[0] == point_labels[0]).sum() / point_labels[0].shape[0]}
        pprint(metric_dict)

        model.train()
        batch_dict = next(iter(train_loader))
        load_data_to_gpu(batch_dict)
        ret_dict, tb_dict, disp_dict = model(batch_dict)
        

if __name__ == '__main__':
    unittest.main()