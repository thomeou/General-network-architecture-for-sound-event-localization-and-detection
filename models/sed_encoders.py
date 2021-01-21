import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from models.model_utils import ConvBlock, init_layer, _ResNet3, _ResNet, _ResnetBasicBlock


class PannCnn14L6(nn.Module):
    """
    Derived from PANN CNN14 network. PannCnn14L6 has 6 CNN layers (3 convblock)
    """
    def __init__(self, n_input_channels: int = 1, p_dropout: float = 0.2, pretrained: bool = False, **kwargs):
        """
        :param n_input_channels: Number of input channels.
        :param p_dropout: Dropout probability.
        :param pretrained: If True, load pretrained model.
        """
        super().__init__()
        self.n_input_channels = n_input_channels
        self.p_dropout = p_dropout
        self.n_output_channels = 256
        self.time_downsample_ratio = 8
        self.freq_downsample_ratio = 8

        self.conv_block1 = ConvBlock(in_channels=n_input_channels, out_channels=64)
        self.conv_block2 = ConvBlock(in_channels=64, out_channels=128)
        self.conv_block3 = ConvBlock(in_channels=128, out_channels=256)

        # Load pretrained model
        self.load_pretrained_weight(pretrained=pretrained)

    def load_pretrained_weight(self, pretrained: bool = False):
        logger = logging.getLogger('lightning')
        pretrained_path = '../pretrained_models/Cnn14_DecisionLevelAtt_mAP=0.425.pth'
        if pretrained:
            checkpoint = torch.load(pretrained_path, map_location=lambda storage, loc: storage)
            try:
                self.load_state_dict(checkpoint['model'], strict=False)
                logger.info('Load pretrained weights from checkpoint {}.'.format(pretrained_path))
            except:
                logger.info('WARNING: Coud not load pretrained weights from checkpoint {}.'.format(pretrained_path))

    def forward(self, x):
        """
        Input x: (batch_size, n_channels, n_timesteps/n_frames, n_features/n_freqs)
        """
        x = self.conv_block1(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=self.p_dropout, training=self.training)
        x = self.conv_block2(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=self.p_dropout, training=self.training)
        x = self.conv_block3(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=self.p_dropout, training=self.training)

        return x

    @property
    def count_number_of_params(self):
        n_params = sum([param.numel() for param in self.parameters()])
        n_trainable_params = sum(param.numel() for param in self.parameters() if param.requires_grad)
        return n_params, n_trainable_params


class PannCnn14L6F64(nn.Module):
    """
    Derived from PANN CNN14 network. PannCnn14L6 has 6 CNN layers (3 convblock)
    """
    def __init__(self, n_input_channels: int = 1, p_dropout: float = 0.2, pretrained: bool = False, **kwargs):
        """
        :param n_input_channels: Number of input channels.
        :param p_dropout: Dropout probability.
        :param pretrained: If True, load pretrained model.
        """
        super().__init__()
        self.n_input_channels = n_input_channels
        self.p_dropout = p_dropout
        self.n_output_channels = 256
        self.time_downsample_ratio = 8
        self.freq_downsample_ratio = 64

        self.conv_block1 = ConvBlock(in_channels=n_input_channels, out_channels=64)
        self.conv_block2 = ConvBlock(in_channels=64, out_channels=128)
        self.conv_block3 = ConvBlock(in_channels=128, out_channels=256)

        # Load pretrained model
        self.load_pretrained_weight(pretrained=pretrained)

    def load_pretrained_weight(self, pretrained: bool = False):
        logger = logging.getLogger('lightning')
        pretrained_path = '../pretrained_models/Cnn14_DecisionLevelAtt_mAP=0.425.pth'
        if pretrained:
            checkpoint = torch.load(pretrained_path, map_location=lambda storage, loc: storage)
            try:
                self.load_state_dict(checkpoint['model'], strict=False)
                logger.info('Load pretrained weights from checkpoint {}.'.format(pretrained_path))
            except:
                logger.info('WARNING: Coud not load pretrained weights from checkpoint {}.'.format(pretrained_path))

    def forward(self, x):
        """
        Input x: (batch_size, n_channels, n_timesteps/n_frames, n_features/n_freqs)
        """
        x = self.conv_block1(x, pool_size=(2, 4), pool_type='avg')
        x = F.dropout(x, p=self.p_dropout, training=self.training)
        x = self.conv_block2(x, pool_size=(2, 4), pool_type='avg')
        x = F.dropout(x, p=self.p_dropout, training=self.training)
        x = self.conv_block3(x, pool_size=(2, 4), pool_type='avg')
        x = F.dropout(x, p=self.p_dropout, training=self.training)

        return x

    @property
    def count_number_of_params(self):
        n_params = sum([param.numel() for param in self.parameters()])
        n_trainable_params = sum(param.numel() for param in self.parameters() if param.requires_grad)
        return n_params, n_trainable_params


class PannCnn14L8(nn.Module):
    """
    Derived from PANN CNN14 network. PannCnn14L8 has 8 CNN layers (4 convblock)
    """
    def __init__(self, n_input_channels: int = 1, p_dropout: float = 0.2, pretrained: bool = False, **kwargs):
        """
        :param n_input_channels: Number of input channels.
        :param p_dropout: Dropout probability.
        :param pretrained: If True, load pretrained model.
        """
        super().__init__()
        self.n_input_channels = n_input_channels
        self.p_dropout = p_dropout
        self.n_output_channels = 512
        self.time_downsample_ratio = 16
        self.freq_downsample_ratio = 16

        self.conv_block1 = ConvBlock(in_channels=n_input_channels, out_channels=64)
        self.conv_block2 = ConvBlock(in_channels=64, out_channels=128)
        self.conv_block3 = ConvBlock(in_channels=128, out_channels=256)
        self.conv_block4 = ConvBlock(in_channels=256, out_channels=512)

        # Load pretrained model
        self.load_pretrained_weight(pretrained=pretrained)

    def load_pretrained_weight(self, pretrained: bool = False):
        logger = logging.getLogger('lightning')
        pretrained_path = '../pretrained_models/Cnn14_DecisionLevelAtt_mAP=0.425.pth'
        if pretrained:
            checkpoint = torch.load(pretrained_path, map_location=lambda storage, loc: storage)
            try:
                self.load_state_dict(checkpoint['model'], strict=False)
                logger.info('Load pretrained weights from checkpoint {}.'.format(pretrained_path))
            except:
                logger.info('WARNING: Coud not load pretrained weights from checkpoint {}.'.format(pretrained_path))

    def forward(self, x):
        """
        Input x: (batch_size, n_channels, n_timesteps/n_frames, n_features/n_freqs)
        """
        x = self.conv_block1(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=self.p_dropout, training=self.training)
        x = self.conv_block2(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=self.p_dropout, training=self.training)
        x = self.conv_block3(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=self.p_dropout, training=self.training)
        x = self.conv_block4(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=self.p_dropout, training=self.training)

        return x

    @property
    def count_number_of_params(self):
        n_params = sum([param.numel() for param in self.parameters()])
        n_trainable_params = sum(param.numel() for param in self.parameters() if param.requires_grad)
        return n_params, n_trainable_params


if __name__ == '__main__':
    encoder = PannCnn14L8()
    print(encoder.count_number_of_params)
    print(encoder)
