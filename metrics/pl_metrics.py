"""
This module wrap the metrics inside pytorch lightning metrics so that we can compute running metrics inside
pytorch lightning module
"""
import math

import numpy as np
import torch
from pytorch_lightning.metrics import Metric


class SedMetrics(Metric):
    """
    Rewrite class SELDMetrics in evaluation_metrics.py for Sed evaluation.
    """
    def __init__(self, nb_frames_1s: int, dist_sync_on_step=False):
        """
        :param nb_frames_1s: Number of frames per second. Often this is the label frame rate
        """
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.block_size = nb_frames_1s
        self.eps = 1e-8
        self.add_state('S', default=torch.tensor(0.0, dtype=torch.float32), dist_reduce_fx="sum")  # dist_reduce_fx is used for distributed training
        self.add_state('D', default=torch.tensor(0.0, dtype=torch.float32), dist_reduce_fx="sum")
        self.add_state('I', default=torch.tensor(0.0, dtype=torch.float32), dist_reduce_fx="sum")
        self.add_state('TP', default=torch.tensor(0.0, dtype=torch.float32), dist_reduce_fx="sum")
        self.add_state('Nref', default=torch.tensor(0.0, dtype=torch.float32), dist_reduce_fx="sum")
        self.add_state('Nsys', default=torch.tensor(0.0, dtype=torch.float32), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        """
        :param preds: (batch_size, n_timesteps, n_classes) or (n_timesteps, n_classes)
        :param target: (batch_size, n_timesteps, n_classes) or (n_timesteps, n_classes)
        """
        assert preds.shape == target.shape
        if preds.ndim == 3:
            if preds.shape[0] == 1:
                preds = torch.squeeze(preds, dim=0)
                target = torch.squeeze(target, dim=0)
            else:
                preds = torch.reshape(preds, (preds.shape[0] * preds.shape[1], -1))
                target = torch.reshape(target, (target.shape[0] * target.shape[1], -1))
        S, D, I = er_overall_1sec(preds, target, self.block_size)
        TP, Nref, Nsys = f1_overall_1sec(preds, target, self.block_size)
        self.S += S
        self.D += D
        self.I += I
        self.TP += TP
        self.Nref += Nref
        self.Nsys += Nsys

    def compute(self):
        if self.Nref == 0:
            self.Nref = self.Nref + 1.0
        ER = (self.S + self.D + self.I) / self.Nref
        prec = self.TP / (self.Nsys + self.eps)
        recall = self.TP / (self.Nref + self.eps)
        F = 2 * prec * recall / (prec + recall + self.eps)
        return ER, F


def f1_overall_1sec(O, T, block_size):
    """
    Legacy code, copied from SELD github repo. To compute F1 for SED metrics.
    :param O: predictions
    :param T: target
    :param block_size: number of frames per 1 s.
    :return:
    """
    new_size = int(math.ceil(float(O.shape[0]) / block_size))
    O_block = torch.zeros((new_size, O.shape[1]), dtype=torch.float32)
    T_block = torch.zeros((new_size, O.shape[1]), dtype=torch.float32)
    for i in range(0, new_size):
        O_block[i, :], _ = torch.max(O[int(i * block_size):int(i * block_size + block_size - 1), :], dim=0)
        T_block[i, :], _ = torch.max(T[int(i * block_size):int(i * block_size + block_size - 1), :], dim=0)
    TP = ((2 * T_block - O_block) == 1).sum()
    Nref, Nsys = T_block.sum(), O_block.sum()
    return TP, Nref, Nsys


def er_overall_1sec(O, T, block_size):
    """
    # TODO combine er_overall_1sec with f1_overall_1sec
    Legacy code, copied from SELD github repo. To compute error rate for SED metrics.
    :param O: predictions
    :param T: target
    :param block_size: number of frames per 1 s.
    """
    new_size = int(math.ceil(float(O.shape[0]) / block_size))
    O_block = torch.zeros((new_size, O.shape[1]))
    T_block = torch.zeros((new_size, O.shape[1]))
    for i in range(0, new_size):
        O_block[i, :], _ = torch.max(O[int(i * block_size):int(i * block_size + block_size - 1), :], dim=0)
        T_block[i, :], _ = torch.max(T[int(i * block_size):int(i * block_size + block_size - 1), :], dim=0)
    FP = torch.logical_and(T_block == 0, O_block == 1).sum(1)
    FN = torch.logical_and(T_block == 1, O_block == 0).sum(1)
    S = torch.minimum(FP, FN).sum()
    D = torch.maximum(torch.tensor(0), FN - FP).sum()
    I = torch.maximum(torch.tensor(0), FP - FN).sum()
    return S, D, I
