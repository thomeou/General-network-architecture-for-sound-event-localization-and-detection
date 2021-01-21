import fire
import logging
import os

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

# from models.rfcx import Rfcx
from utilities.builder_utils import build_database, build_datamodule, build_model, build_task
from utilities.experiments_utils import manage_experiments
from utilities.learning_utils import LearningRateScheduler, MyLoggingCallback


def train(exp_config: str = './configs/sed.yml',
          exp_group_dir: str = '/home/tho_nguyen/Documents/work/seld/outputs/',
          exp_suffix: str = '',
          resume: bool = False,
          empty: bool = False):
    """
    Training script
    :param exp_config: Config file for experiments
    :param exp_group_dir: Parent directory to store all experiment results.
    :param exp_suffix: Experiment suffix.
    :param resume: If true, resume training from the last epoch.
    :param empty: If true, delete all previous data in experiment folder.
    """
    # Load config, create folders, logging
    cfg = manage_experiments(exp_config=exp_config, exp_group_dir=exp_group_dir, exp_suffix=exp_suffix, empty=empty)
    logger = logging.getLogger('lightning')

    # Set random seed for reproducible
    pl.seed_everything(cfg.seed)

    # Resume training
    if resume:
        ckpt_list = [f for f in os.listdir(cfg.dir.model.checkpoint) if f.startswith('epoch') and f.endswith('ckpt')]
        if len(ckpt_list) > 0:
            resume_from_checkpoint = os.path.join(cfg.dir.model.checkpoint, sorted(ckpt_list)[-1])
            logger.info('Found checkpoint to be resume training at {}'.format(resume_from_checkpoint))
        else:
            resume_from_checkpoint = None
    else:
        resume_from_checkpoint = None

    # Load feature database  - will use a builder function build_feature_db to select feature db.
    feature_db = build_database(cfg=cfg)

    # Load data module
    datamodule = build_datamodule(cfg=cfg, feature_db=feature_db)

    # Model checkpoint
    model_checkpoint = ModelCheckpoint(dirpath=cfg.dir.model.checkpoint, filename='{epoch:03d}')

    # Console logger
    console_logger = MyLoggingCallback()

    # Tensorboard logger
    tb_logger = TensorBoardLogger(save_dir=cfg.dir.tb_dir, name='my_model')

    # Build encoder and decoder
    encoder_params = cfg.model.encoder.__dict__
    encoder_kwargs = {'n_input_channels': cfg.data.n_input_channels, **encoder_params}
    encoder = build_model(**encoder_kwargs)
    decoder_params = cfg.model.decoder.__dict__
    decoder_params = {'n_classes': cfg.data.n_classes, 'encoder_output_channels': encoder.n_output_channels,
                      **decoder_params}
    decoder = build_model(**decoder_params)

    # Build Lightning model
    model = build_task(encoder=encoder, decoder=decoder, cfg=cfg)

    # Train
    callback_list = [console_logger, model_checkpoint]
    trainer = pl.Trainer(gpus=torch.cuda.device_count(), resume_from_checkpoint=resume_from_checkpoint,
                         max_epochs=cfg.training.max_epochs, logger=tb_logger, progress_bar_refresh_rate=2,
                         check_val_every_n_epoch=cfg.training.val_interval,
                         log_every_n_steps=100, flush_logs_every_n_steps=200,
                         limit_train_batches=cfg.data.train_fraction, limit_val_batches=cfg.data.val_fraction,
                         callbacks=callback_list)
    trainer.fit(model, datamodule)


if __name__ == '__main__':
    fire.Fire(train)
