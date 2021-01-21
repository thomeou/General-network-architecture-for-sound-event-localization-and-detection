"""
This module include code to perform SED task
"""
import logging
import os
from typing import Tuple

import h5py
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

from metrics.pl_metrics import SedMetrics
from models.model_utils import interpolate_tensor, freeze_model, unfreeze_model


class SedModel(pl.LightningModule):
    def __init__(self, encoder: nn.Module, decoder: nn.Module, sed_threshold: float = 0.3, label_rate: int = 10,
                 encoder_unfreeze_epoch: int = 0, optimizer_name: str = 'Adam', lr: float = 1e-3,
                 output_pred_dir: str = None, submission_filename: str = None, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.encoder = encoder
        self.decoder = decoder
        self.sed_threshold = sed_threshold
        self.label_rate = label_rate
        self.encoder_unfreeze_epoch = encoder_unfreeze_epoch
        self.optimizer_name = optimizer_name
        self.lr = lr
        self.submission_fn = submission_filename
        self.output_pred_dir = output_pred_dir
        self.time_downsample_ratio = self.encoder.time_downsample_ratio
        self.n_classes = self.decoder.n_classes
        self.lit_logger = logging.getLogger('lightning')

        # Metrics
        self.train_sed_metrics = SedMetrics(nb_frames_1s=self.label_rate)
        self.valid_sed_metrics = SedMetrics(nb_frames_1s=self.label_rate)
        self.test_sed_metrics = SedMetrics(nb_frames_1s=self.label_rate)

        # Freeze encoder layer
        self.freeze_encoder()

        # Write submission files
        self.columns = ['frame_idx', 'event', 'azimuth', 'elevation']
        self.submission = pd.DataFrame(columns=self.columns)

        # Write output prediction for test step
        if self.output_pred_dir is not None:
            os.makedirs(self.output_pred_dir, exist_ok=True)

    def freeze_encoder(self):
        if self.encoder_unfreeze_epoch == -1 or self.encoder_unfreeze_epoch > 0:
            freeze_model(self.encoder)

    def forward(self, x):
        """
        x: (batch_size, n_channels, n_timesteps (n_frames), n_features).
        """
        x = self.encoder(x)  # (batch_size, n_channels, n_timesteps, n_features)
        output_dict = self.decoder(x)  # (batch_size, n_timesteps, n_classes)
        # output_dict = {
        #     "event_frame_logit": event_frame_logit,
        # }
        return output_dict

    def common_step(self, batch_data):
        x, y_sed, _, _ = batch_data
        y_sed = interpolate_tensor(y_sed, ratio=1.0 / self.time_downsample_ratio)  # to match output dimension
        target_dict = {
            'event_frame_gt': y_sed,
        }
        pred_dict = self.forward(x)
        event_frame_output = (torch.sigmoid(pred_dict['event_frame_logit']) > self.sed_threshold).type(torch.float32)
        return target_dict, pred_dict, event_frame_output

    def training_step(self, train_batch, batch_idx):
        target_dict, pred_dict, event_frame_output = self.common_step(train_batch)
        loss = self.compute_loss(target_dict=target_dict, pred_dict=pred_dict)
        self.train_sed_metrics(event_frame_output, target_dict['event_frame_gt'])
        # logging
        self.log('trl', loss, prog_bar=True, logger=True)
        training_step_outputs = {'loss': loss, 'event_frame_logit': pred_dict['event_frame_logit'],
                                 'event_frame_gt': target_dict['event_frame_gt']}
        return training_step_outputs

    def training_epoch_end(self, training_step_outputs):
        # Unfreeze encoder
        if 0 < self.encoder_unfreeze_epoch == self.current_epoch:
            unfreeze_model(self.encoder)
            self.lit_logger.info('Unfreezing encoder at epoch: {}'.format(self.current_epoch))
        # compute running metric
        ER, F1 = self.train_sed_metrics.compute()
        sed_error = (ER + 1 - F1)/2
        self.log('trER', ER)
        self.log('trF1', F1)
        self.log('trSedE', sed_error)
        self.lit_logger.info('Epoch {} - Training - ER: {:.4f} - F1: {:.4f} - SED error: {:.4f}'.format(
            self.current_epoch, ER, F1, sed_error))

    def validation_step(self, val_batch, batch_idx):
        target_dict, pred_dict, event_frame_output = self.common_step(val_batch)
        loss = self.compute_loss(target_dict=target_dict, pred_dict=pred_dict)
        self.valid_sed_metrics(event_frame_output, target_dict['event_frame_gt'])
        # logging
        self.log('vall', loss, prog_bar=True, logger=True)
        return None

    def validation_epoch_end(self, validation_step_outputs):
        # compute running metric for SED
        ER, F1 = self.valid_sed_metrics.compute()
        sed_error = (ER + 1 - F1) / 2
        self.log('valER', ER)
        self.log('valF1', F1)
        self.log('valSedE', sed_error)
        self.lit_logger.info('Epoch {} - Validation - ER: {:.4f} - F1: {:.4f} - SED error: {:.4f}'.format(
            self.current_epoch, ER, F1, sed_error))

    def test_step(self, test_batch, batch_idx):
        target_dict, pred_dict, event_frame_output = self.common_step(test_batch)
        self.test_sed_metrics(event_frame_output, target_dict['event_frame_gt'])
        filenames = test_batch[-1]
        # TODO submission file
        # # add output to submission dataframe
        # self.append_output_prediction(y_pred=y_pred_file, filenames=filenames)
        # Write output prediction
        if self.output_pred_dir:
            h5_filename = os.path.join(self.output_pred_dir, filenames[0] + '.h5')
            event_frame_pred = torch.sigmoid(pred_dict['event_frame_logit']).detach().cpu().numpy()
            event_frame_gt = target_dict['event_frame_gt'].detach().cpu.numpy()
            with h5py.File(h5_filename, 'w') as hf:
                hf.create_dataset('event_frame_pred', data=event_frame_pred, dtype=np.float32)
                hf.create_dataset('event_frame_gt', data=event_frame_gt, dtype=np.float32)
                hf.create_dataset('time_downsample_ratio', data=self.time_downsample_ratio, dtype=np.float32)
        return None

    def test_epoch_end(self, test_step_outputs):
        ER, F1 = self.test_sed_metrics.compute()
        sed_error = (ER + 1 - F1) / 2
        self.log('testER', ER)
        self.log('testF1', F1)
        self.log('testSedE', sed_error)
        self.lit_logger.info('Epoch {} - Test - ER: {:.4f} - F1: {:.4f} - SED error: {:.4f}'.format(
            self.current_epoch, ER, F1, sed_error))
        # # TODO
        # # write to output file
        # self.write_output_submission()

    @staticmethod
    def compute_loss(target_dict, pred_dict):
        # Event frame loss
        sed_loss = F.binary_cross_entropy_with_logits(input=pred_dict['event_frame_logit'],
                                                      target=target_dict['event_frame'])
        return sed_loss

    def configure_optimizers(self):
        """
        Pytorch lightning hook
        """
        if self.optimizer_name in ['Adam', 'adam']:
            optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        elif self.optimizer_name in ['AdamW', 'Adamw', 'adamw']:
            optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        else:
            raise NotImplementedError('Optimizer {} is not implemented!'.format(self.optimizer_name))
        return optimizer

    def append_output_prediction(self, y_pred, filenames):
        assert len(set(filenames)) == 1, 'Test batch contains different audio files.'
        if self.submission_fn is not None:
            filename = filenames[0]
            prediction = dict(zip(self.columns[1:], y_pred.cpu().numpy()[0]))
            prediction['recording_id'] = filename
            self.submission = self.submission.append(prediction, ignore_index=True)

    def write_output_submission(self):
        if self.submission_fn is not None:
            self.submission.to_csv(self.submission_fn, index=False)

