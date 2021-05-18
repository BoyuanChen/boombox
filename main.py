
import os
import sys
import yaml
import torch
import pprint
from munch import munchify
from models import SoundBoxModel
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint


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
                          if_test=False,
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

    # define callback for selecting checkpoints during training
    checkpoint_callback = ModelCheckpoint(
        filepath=log_dir + "/lightning_logs/checkpoints/{epoch}_{iou_score}",
        verbose=True,
        monitor='iou_score',
        mode='max',
        prefix='')

    # define trainer
    trainer = Trainer(gpus=cfg.num_gpus,
                      max_epochs=cfg.epochs,
                      deterministic=True,
                      accelerator='ddp',
                      amp_backend='native',
                      default_root_dir=log_dir,
                      val_check_interval=1.0,
                      checkpoint_callback=checkpoint_callback)

    trainer.fit(model)



if __name__ == '__main__':
    main()
