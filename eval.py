

import os
import sys
import glob
import yaml
import torch
import pprint
from munch import munchify
from models import SoundBoxModel
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import TensorBoardLogger


def load_config(filepath):
    with open(filepath, 'r') as stream:
        try:
            trainer_params = yaml.safe_load(stream)
            return trainer_params
        except yaml.YAMLError as exc:
            print(exc)

def seed(cfg):
    torch.manual_seed(cfg.seed)
    if cfg.if_cuda:
        torch.cuda.manual_seed(cfg.seed)


def main():
    config_filepath = str(sys.argv[1])
    checkpoint_filepath = str(sys.argv[2])
    checkpoint_filepath = glob.glob(os.path.join(checkpoint_filepath, '*.ckpt'))[0]
    cfg = load_config(filepath=config_filepath)
    pprint.pprint(cfg)
    cfg = munchify(cfg)
    seed(cfg)
    seed_everything(cfg.seed)

    log_dir = '_'.join([cfg.log_dir,
                        str(cfg.if_sound),
                        str(cfg.if_vision),
                        str(cfg.if_depth),
                        cfg.depth_representation,
                        cfg.model_name,
                        str(cfg.if_all_input_data),
                        cfg.output_representation,
                        str(cfg.seed)])

    model = SoundBoxModel(lr=cfg.lr,
                          seed=cfg.seed,
                          if_cuda=cfg.if_cuda,
                          if_test=True,
                          gamma=cfg.gamma,
                          log_dir=log_dir,
                          train_batch=cfg.train_batch,
                          val_batch=cfg.val_batch,
                          test_batch=cfg.test_batch,
                          num_workers=cfg.num_workers,
                          in_channels=cfg.in_channels,
                          model_name=cfg.model_name,
                          num_branches=cfg.num_branches,
                          branches_in_channels=cfg.branches_in_channels,
                          data_filepath=cfg.data_filepath,
                          shapes=cfg.shapes,
                          if_sound=cfg.if_sound,
                          if_vision=cfg.if_vision,
                          if_depth=cfg.if_depth,
                          if_all_input_data=cfg.if_all_input_data,
                          depth_representation=cfg.depth_representation,
                          output_representation=cfg.output_representation,
                          lr_schedule=cfg.schedule,
                          test_hsv_threshold_lst=cfg.test_hsv_threshold_lst)

    ckpt = torch.load(checkpoint_filepath)
    model.load_state_dict(ckpt['state_dict'])
    model.eval()
    model.freeze()

    trainer = Trainer(gpus=1,
                      deterministic=True,
                      amp_backend='native',
                      default_root_dir=log_dir,
                      val_check_interval=1.0)

    trainer.test(model)



if __name__ == '__main__':
    main()