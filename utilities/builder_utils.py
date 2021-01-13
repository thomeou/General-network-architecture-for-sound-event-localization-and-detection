"""
This modules consists code to select different components for
    feature_database
    models
"""
import logging

import torch
import torch.nn as nn

import models
from dataset.database import SedDoaDatabase
from dataset.datamodule import SedDoaDataModule


def build_database(cfg):
    """
    Function to select database according to task
    :param cfg: Experiment config
    """
    if cfg.task in ['sed', 'SED', 'doa', 'DOA']:
        feature_db = SedDoaDatabase(feature_root_dir=cfg.feature_dir, gt_meta_root_dir=cfg.split_meta_dir,
                                    audio_format=cfg.data.audio_format, n_classes=cfg.data.n_classes, fs=cfg.data.fs,
                                    n_fft=cfg.data.n_fft, hop_len=cfg.data.hop_len, label_rate=cfg.data.label_rate,
                                    train_chunk_len_s=cfg.data.train_chunk_len_s,
                                    train_chunk_hop_len_s=cfg.data.train_chunk_hop_len_s,
                                    test_chunk_len_s=cfg.data.test_chunk_len_s,
                                    test_chunk_hop_len_s=cfg.data.test_chunk_hop_len_s,
                                    scaler_type=cfg.data.scaler_type)
    elif cfg.task in ['seld', 'SELD']:
        pass
    else:
        raise NotImplementedError('task {} is not implemented'.format(cfg.task))

    return feature_db


def build_datamodule(cfg, feature_db):
    """
    Function to select pytorch lightning datamodule according to different tasks.
    :param cfg: Experiment config.
    :param feature_db: Feature database.
    """
    if cfg.task in ['sed', 'SED', 'doa', 'DOA']:
        datamodule = SedDoaDataModule(feature_db=feature_db, split_meta_dir=cfg.split_meta_dir, mode=cfg.mode,
                                      train_batch_size=cfg.training.train_batch_size,
                                       val_batch_size=cfg.training.val_batch_size)

    elif cfg.task in ['seld', 'SELD']:
        pass
    else:
        raise NotImplementedError('task {} is not implemented'.format(cfg.task))

    return datamodule



def build_model(model_name, pretrained_path: str = None, load_type: str = 'partial', cp_key: str = 'model_state_dict',
                **kwargs) -> nn.Module:
    """
    Function to select model and load pretrained weight if available
    :param model_name: Name of model.
    :param pretrained_path: Path to pretrained state dict.
    :param load_type: if 'strict': load everything, if 'partial': load matched state dict
    :param cp_key: to access the checkpoint:
        cp_key = 'model_state_dict': for normal pytorch model saved with
            torch.save(model.state_dict(), PATH)' (not verifed)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
                ...
                }, PATH)
        cp_key = 'model': to load PANNs model
        cp_key = 'state_dict': pytorch lightning model.
    :return: Pytorch nn.Module.
    """
    # Load model:
    model = models.__dict__[model_name](**kwargs)
    logger = logging.getLogger('lightning')
    logger.info('Finish loading model {}.'.format(model_name))

    # Load pretrained weights
    if pretrained_path:
        checkpoint = torch.load(pretrained_path, map_location=lambda storage, loc: storage)
        try:
            if load_type == 'strict':
                model.load_state_dict(checkpoint[cp_key])
            else:
                model.load_state_dict(checkpoint[cp_key], strict=False)

            logger.info('Load pretrained weights from checkpoint {}.'.format(pretrained_path))
        except:
            logger.info('WARNING: Coud not load pretrained weights from checkpoint {}.'.format(pretrained_path))
    else:
        logger.info('No loading pretrained weights.')

    return model


