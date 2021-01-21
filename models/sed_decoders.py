"""
This modules include decoders for sound classification.
"""
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.model_utils import AttBlock, ConvBlock, init_layer, init_gru, PositionalEncoding


class SedDecoder(nn.Module):
    """
    Decoder for SED
    """
    def __init__(self, encoder_output_channels: int, n_classes: int = 14, freq_pool: str = 'avg',
                 decoder_type: str = 'gru', **kwargs):
        """
        :param encoder_output_channels: Number of output channels/filter of encoder.
        :param n_classes: Number of classes.
        :param freq_pool: Type of frequency pooling. Choices are:
            'avg': average pooling
            'max': max pooling
            'avg_max': add average and max pooling
        :param decoder_type: Choices are:
            'gru':
        """
        super().__init__()
        self.decoder_input_size = encoder_output_channels
        self.n_classes = n_classes
        self.freq_pool = freq_pool
        self.decoder_type = decoder_type

        if self.decoder_type == 'gru':
            self.gru_hidden_size = self.decoder_input_size//2
            self.fc_size = self.gru_hidden_size * 2

            self.gru = nn.GRU(input_size=self.decoder_input_size, hidden_size=self.gru_hidden_size, num_layers=1,
                              batch_first=True, bidirectional=True)
        else:
            raise NotImplementedError('decoder type {} is not implemented'.format(self.decoder_type))

        self.event_fc = nn.Linear(self.fc_size, self.n_classes, bias=True)

        self.init_weights()

    def init_weights(self):
        if self.decoder_type == 'gru':
            init_gru(self.gru)
        init_layer(self.event_fc)

    def forward(self, x):
        """
        Input x: (batch_size, n_channels, n_timesteps/n_frames (downsampled), n_features/n_freqs (downsampled)
        """
        if self.freq_pool == 'avg':
            x = torch.mean(x, dim=3)
        elif self.freq_pool == 'max':
            (x, _) = torch.max(x, dim=3)
        elif self.freq_pool == 'avg_max':
            x1 = torch.mean(x, dim=3)
            (x2, _) = torch.max(x, dim=3)
            x = x1 + x
        else:
            raise ValueError('freq pooling {} is not implemented'.format(self.freq_pool))
        '''(batch_size, feature_maps, time_steps)'''

        if self.decoder_type == 'gru':
            x = x.transpose(1, 2)
            ''' (batch_size, time_steps, feature_maps):'''
            (x, _) = self.gru(x)
        else:
            raise NotImplementedError('decoder type {} is not implemented'.format(self.decoder_type))

        x = F.dropout(x, p=0.2, training=self.training)
        event_frame_logit = self.event_fc(x)
        '''(batch_size, time_steps, class_num)'''

        output = {
            'event_frame_logit': event_frame_logit
        }

        return output
