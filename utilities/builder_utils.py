"""
This modules consists code to select different components for
    feature_database
    models
"""
import logging

import pytorch_lightning as pl
import torch
import torch.nn as nn

import models
from dataset.database import SedDoaDatabase
from dataset.datamodule import SedDoaDataModule
from models.sed_models import SedModel


def build_database(cfg):
    """
    Function to select database according to task
    :param cfg: Experiment config
    """
    if cfg.task in ['sed', 'SED', 'doa', 'DOA']:
        feature_db = SedDoaDatabase(feature_root_dir=cfg.feature_root_dir, gt_meta_root_dir=cfg.gt_meta_root_dir,
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


def build_model(name: str, **kwargs) -> nn.Module:
    """
    Build encoder.
    :param name: Name of the encoder.
    :return: encoder model
    """
    logger = logging.getLogger('lightning')
    # Load model:
    model = models.__dict__[name](**kwargs)
    logger.info('Finish loading model {}.'.format(name))

    return model


def build_task(encoder, decoder, cfg, **kwargs) -> pl.LightningModule:
    """
    Build task
    :param encoder:
    :param decoder:
    :param cfg:
    :return: Lightning module
    """
    if cfg.task in ['sed', 'Sed', 'SED']:
        model = SedModel(encoder=encoder, decoder=decoder, encoder_unfreeze_epoch=cfg.model.encoder.unfreeze_epoch,
                         sed_threshold=cfg.data.sed_threshold, label_rate=cfg.data.label_rate,
                         optimizer_name=cfg.training.optimizer)
    else:
        pass

    return model

