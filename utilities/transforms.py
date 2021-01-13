"""
This module includes code to do data augmentation in STFT domain on numpy array:
    1. random volume
    2. random cutout
    3. spec augment
    4. freq shift
    5. TTA: test time augmentation
"""
import random
from typing import Tuple

import numpy as np
import torch


class ComposeTransformNp:
    """
    Compose a list of data augmentation on numpy array.
    """
    def __init__(self, transforms: list):
        self.transforms = transforms

    def __call__(self, x: np.ndarray):
        for transform in self.transforms:
            x = transform(x)
        return x


class DataAugSpectrogramNp:
    """
    Base class for data augmentation for audio spectrogram of numpy array. This class does not alter label
    """
    def __init__(self, always_apply: bool = False, p: float = 0.5):
        self.always_apply = always_apply
        self.p = p

    def __call__(self, x: np.ndarray):
        if self.always_apply:
            return self.apply(x)
        else:
            if np.random.rand() < self.p:
                return self.apply(x)
            else:
                return x

    def apply(self, x: np.ndarray):
        raise NotImplementedError


class RandomCutoutNp(DataAugSpectrogramNp):
    """
    This data augmentation randomly cutout a rectangular area from the input image. Tested.
    """
    def __init__(self, always_apply: bool = False, p: float = 0.5, image_aspect_ratio: float = 1,
                 random_value: float = None):
        """
        :param always_apply: If True, always apply transform.
        :param p: If always_apply is false, p is the probability to apply transform.
        :param image_aspect_ratio: height/width ratio. For spectrogram: n_time_steps/ n_features.
        :param random_value: random value to fill in the cutout area. If None, randomly fill the cutout area with value
            between min and max of input.
        """
        super().__init__(always_apply, p)
        self.random_value = random_value
        # Params: s: area, r: height/width ratio.
        self.s_l = 0.02
        self.s_h = 0.3
        self.r_1 = 0.3
        self.r_2 = 1 / 0.3
        if image_aspect_ratio > 1:
            self.r_1 = self.r_1 * image_aspect_ratio
        elif image_aspect_ratio < 1:
            self.r_2 = self.r_2 * image_aspect_ratio

    def apply(self, x: np.ndarray) -> np.ndarray:
        """
        :param x: <(n_channels, n_time_steps, n_features) or (n_time_steps, n_features)>: input spectrogram.
        :return: random cutout x
        """
        # get image size
        image_dim = x.ndim
        img_h = x.shape[-2]  # time frame dimension
        img_w = x.shape[-1]  # feature dimension
        min_value = np.min(x)
        max_value = np.max(x)
        # Initialize output
        output_img = x.copy()
        # random erase
        s = np.random.uniform(self.s_l, self.s_h) * img_h * img_w
        r = np.random.uniform(self.r_1, self.r_2)
        w = np.min((int(np.sqrt(s / r)), img_w - 1))
        h = np.min((int(np.sqrt(s * r)), img_h - 1))
        left = np.random.randint(0, img_w - w)
        top = np.random.randint(0, img_h - h)
        if self.random_value is None:
            c = np.random.uniform(min_value, max_value)
        else:
            c = self.random_value
        if image_dim == 2:
            output_img[top:top + h, left:left + w] = c
        else:
            output_img[:, top:top + h, left:left + w] = c

        return output_img


class SpecAugmentNp(DataAugSpectrogramNp):
    """
    This data augmentation randomly remove horizontal or vertical strips from image. Tested
    """
    def __init__(self, always_apply: bool = False, p: float = 0.5, time_max_width: int = None,
                 freq_max_width: int = None, n_time_stripes: int = 1, n_freq_stripes: int = 1):
        """
        :param always_apply: If True, always apply transform.
        :param p: If always_apply is false, p is the probability to apply transform.
        :param time_max_width: maximum time width to remove.
        :param freq_max_width: maximum freq width to remove.
        :param n_time_stripes: number of time stripes to remove.
        :param n_freq_stripes: number of freq stripes to remove.
        """
        super().__init__(always_apply, p)
        self.time_max_width = time_max_width
        self.freq_max_width = freq_max_width
        self.n_time_stripes = n_time_stripes
        self.n_freq_stripes = n_freq_stripes

    def apply(self, x: np.ndarray) -> np.ndarray:
        """
        :param x: <(n_channels, n_time_steps, n_features)>: input spectrogram.
        :return: augmented spectrogram.
        """
        assert x.ndim == 3, 'Error: dimension of input spectrogram is not 3!'
        n_frames = x.shape[1]
        n_freqs = x.shape[2]
        min_value = np.min(x)
        max_value = np.max(x)
        if self.time_max_width is None:
            time_max_width = int(0.15 * n_frames)
        else:
            time_max_width = self.time_max_width
        time_max_width = np.max((1, time_max_width))
        if self.freq_max_width is None:
            freq_max_width = int(0.2 * n_freqs)
        else:
            freq_max_width = self.freq_max_width
        freq_max_width = np.max((1, freq_max_width))

        new_spec = x.copy()

        for i in np.arange(self.n_time_stripes):
            dur = np.random.randint(1, time_max_width, 1)[0]
            start_idx = np.random.randint(0, n_frames - dur, 1)[0]
            random_value = np.random.uniform(min_value, max_value, 1)
            new_spec[:, start_idx:start_idx + dur, :] = random_value

        for i in np.arange(self.n_freq_stripes):
            dur = np.random.randint(1, freq_max_width, 1)[0]
            start_idx = np.random.randint(0, n_freqs - dur, 1)[0]
            random_value = np.random.uniform(min_value, max_value, 1)
            new_spec[:, :, start_idx:start_idx + dur] = random_value

        return new_spec


class RandomCutoutHoleNp(DataAugSpectrogramNp):
    """
    This data augmentation randomly cutout a few small holes in the spectrogram. Tested.
    """
    def __init__(self, always_apply: bool = False, p: float = 0.5, n_max_holes: int = 8, max_h_size: int = 8,
                 max_w_size: int = 8, filled_value: float = None):
        """
        :param always_apply: If True, always apply transform.
        :param p: If always_apply is false, p is the probability to apply transform.
        :param n_max_holes: Maximum number of holes to cutout.
        :param max_h_size: Maximum time frames of the cutout holes.
        :param max_w_size: Maximum freq bands of the cutout holes.
        :param filled_value: random value to fill in the cutout area. If None, randomly fill the cutout area with value
            between min and max of input.
        """
        super().__init__(always_apply, p)
        self.n_max_holes = n_max_holes
        self.max_h_size = np.max((max_h_size, 5))
        self.max_w_size = np.max((max_w_size, 5))
        self.filled_value = filled_value

    def apply(self, x: np.ndarray):
        """
        :param x: <(n_channels, n_time_steps, n_features)>: input spectrogram.
        :return: augmented spectrogram.
        """
        assert x.ndim == 3, 'Error: dimension of input spectrogram is not 3!'
        img_h = x.shape[-2]  # time frame dimension
        img_w = x.shape[-1]  # feature dimension
        min_value = np.min(x)
        max_value = np.max(x)
        new_spec = x.copy()
        # n_cutout_holes = np.random.randint(1, self.n_max_holes, 1)[0]
        n_cutout_holes = self.n_max_holes
        for ihole in np.arange(n_cutout_holes):
            # w = np.random.randint(4, self.max_w_size, 1)[0]
            # h = np.random.randint(4, self.max_h_size, 1)[0]
            w = self.max_w_size
            h = self.max_h_size
            left = np.random.randint(0, img_w - w)
            top = np.random.randint(0, img_h - h)
            if self.filled_value is None:
                new_spec[:, top:top + h, left:left + w] = np.random.uniform(min_value, max_value)
            else:
                new_spec[:, top:top + h, left:left + w] = self.filled_value

        return new_spec


class CompositeCutout(DataAugSpectrogramNp):
    """
    This data augmentation combine Random cutout, specaugment, cutout hole.
    """
    def __init__(self, always_apply: bool = False, p: float = 0.5, image_aspect_ratio: float = 1):
        super().__init__(always_apply, p)
        self.random_cutout = RandomCutoutNp(always_apply=True, image_aspect_ratio=image_aspect_ratio)
        self.spec_augment = SpecAugmentNp(always_apply=True)
        self.random_cutout_hole = RandomCutoutHoleNp(always_apply=True)

    def apply(self, x: np.ndarray):
        choice = np.random.randint(0, 2, 1)[0]
        if choice == 0:
            return self.random_cutout(x)
        elif choice == 1:
            return self.spec_augment(x)


class RandomFlipLeftRightNp(DataAugSpectrogramNp):
    """
    This data augmentation randomly flip spectrogram left and right.
    TODO also flip labels
    """
    def __init__(self, always_apply: bool = False, p: float = 0.5):
        super().__init__(always_apply, p)

    def apply(self, x: np.ndarray):
        """
        :param x < np.ndarray (n_channels, n_time_steps, n_features)
        :return:
        """
        new_x = x.copy()
        for ichan in np.arange(x.shape[0]):
            new_x[ichan, :] = np.flip(x[ichan, :], axis=0)
        return new_x


class AdditiveGaussianNoiseNp(DataAugSpectrogramNp):
    """
    This data augmentation add gaussian noise to spectrogram. Assume spectrograms are mean-var normalzied.
    """
    def __init__(self, always_apply: bool = False, p: float = 0.5):
        super().__init__(always_apply, p)

    def apply(self, x: np.ndarray):
        """
        :param x < np.ndarray (n_channels, n_time_steps, n_features)
        """
        n_frames, n_features = x.shape[1], x.shape[2]
        jitter_std = np.random.uniform(0.05, 0.2, 1)
        jitter = np.random.normal(0, jitter_std, size=(n_frames, n_features)).astype(np.float32)
        new_spec = x.copy()
        new_spec = new_spec + jitter
        return new_spec


class MultiplicativeGaussianNoiseNp(DataAugSpectrogramNp):
    """
    This data augmentation multiply gaussian noise to spectrogram. Assume spectrograms are mean-var normalzied.
    """
    def __init__(self, always_apply: bool = False, p: float = 0.5):
        super().__init__(always_apply, p)

    def apply(self, x: np.ndarray):
        """
        :param x < np.ndarray (n_channels, n_time_steps, n_features)
        """
        n_frames, n_features = x.shape[1], x.shape[2]
        jitter_std = np.random.uniform(0.01, 0.1, 1)
        jitter = np.random.normal(0, jitter_std, size=(n_frames, n_features)).astype(np.float32) + 1
        new_spec = x.copy()
        new_spec = new_spec * jitter
        return new_spec


class CosineGaussianNoiseNp(DataAugSpectrogramNp):
    """
    This data augmentation add/multiply Gaussian noise whose power has sinusoidal pattern over time.
    """
    def __init__(self, always_apply: bool = False, p: float = 0.5):
        super().__init__(always_apply, p)

    def apply(self, x: np.ndarray):
        """
        :param x < np.ndarray (n_channels, n_time_steps, n_features)
        """
        n_frames, n_features = x.shape[1], x.shape[2]
        jitter_std = np.random.uniform(0.01, 0.1, 1)
        jitter = np.random.normal(0, jitter_std, size=(n_frames, n_features)).astype(np.float32)
        cosine = np.cos(np.arange(n_frames) / n_frames * np.pi * 2).astype(np.float32)
        jitter = jitter * cosine[:, None] + 1
        new_spec = x.copy()
        new_spec = new_spec * jitter
        return new_spec


class CompositeGaussianNoiseNp(DataAugSpectrogramNp):
    """
    This data augmentation randomly select different method to add gaussian noise to the spectrograms
    """
    def __init__(self, always_apply: bool = False, p: float = 0.5):
        super().__init__(always_apply, p)
        self.agauss = AdditiveGaussianNoiseNp(always_apply=True)
        self.mgauss = MultiplicativeGaussianNoiseNp(always_apply=True)
        self.cgauss = CosineGaussianNoiseNp(always_apply=True)

    def apply(self, x: np.ndarray):
        choice = np.random.randint(0, 2, 1)[0]
        if choice == 0:
            return self.agauss(x)
        elif choice == 1:
            return self.mgauss(x)


class RandomVolumeNp(DataAugSpectrogramNp):
    """
    This data augmentation randomly increase or decrease volume
    Reference from: https://github.com/koukyo1994/kaggle-birdcall-6th-place/blob/master/src/transforms.py
    """
    def __init__(self, always_apply: bool = False, p: float = 0.5, limit=3):
        super().__init__(always_apply, p)
        self.limit = limit

    def apply(self, x: np.ndarray):
        db = np.random.uniform(-self.limit, self.limit)
        new_spec = x.copy()
        new_spec = new_spec * db2float(db, amplitude=False)
        return new_spec


class CosineVolumeNp(DataAugSpectrogramNp):
    """
    This data augmentation change volume in cosine pattern
    Reference from: https://github.com/koukyo1994/kaggle-birdcall-6th-place/blob/master/src/transforms.py
    """
    def __init__(self, always_apply=False, p=0.5, limit=3):
        super().__init__(always_apply, p)
        self.limit = limit

    def apply(self, x: np.ndarray):
        """
        :param x < np.ndarray (n_channels, n_time_steps, n_features)
        """
        db = np.random.uniform(-self.limit, self.limit)
        n_time_steps = x.shape[1]
        cosine = np.cos(np.arange(n_time_steps) / n_time_steps * np.pi * 2)
        dbs = db2float(cosine * db)
        new_spec = x.copy()
        return new_spec * dbs[None, :, None]


def db2float(db: float, amplitude=True):
    """Function to convert dB to float"""
    if amplitude:
        return 10**(db / 20)
    else:
        return 10 ** (db / 10)


class CompositeVolumeNp(DataAugSpectrogramNp):
    """
    This data augmentation randomly select different method to add gaussian noise to the spectrograms
    """
    def __init__(self, always_apply: bool = False, p: float = 0.5):
        super().__init__(always_apply, p)
        self.random_volume = RandomVolumeNp(always_apply=True)
        self.cosine_volume = CosineVolumeNp(always_apply=True)

    def apply(self, x: np.ndarray):
        choice = np.random.randint(0, 2, 1)[0]
        if choice == 0:
            return self.random_volume(x)
        elif choice == 1:
            return self.cosine_volume(x)


class RandomRotateNp(DataAugSpectrogramNp):
    """
    This data augmentation rotate spectrogram along time dimension.
    """
    def __init__(self, always_apply=False, p=0.5, max_length: int = None, direction: str = None):
        super().__init__(always_apply, p)
        self.max_length = max_length
        self.direction = direction

    def apply(self, x: np.ndarray):
        n_channels, n_timesteps, n_features = x.shape
        if self.max_length is None:
            self.max_length = int(n_timesteps * 0.1)
        rotate_len = np.random.randint(1, self.max_length, 1)[0]
        new_spec = x.copy()
        if self.direction is None:
            direction = np.random.choice(['left', 'right'], 1)[0]
        else:
            direction = self.direction
        if direction == 'left':  # rotate left
            new_spec[:, 0:-rotate_len, :] = x[:, rotate_len:, :]
            new_spec[:, -rotate_len:, ] = x[:, 0: rotate_len, :]
        else:  # rotate right
            new_spec[:, rotate_len:, :] = x[:, 0: -rotate_len, :]
            new_spec[:, 0: rotate_len, :] = x[:, -rotate_len:, :]
        return new_spec


class RandomShiftUpDownNp(DataAugSpectrogramNp):
    """
    This data augmentation random shift the spectrogram up or down.
    """
    def __init__(self, always_apply=False, p=0.5, freq_shift_range: int = None, direction: str = None, mode='reflect'):
        super().__init__(always_apply, p)
        self.freq_shift_range = freq_shift_range
        self.direction = direction
        self.mode = mode

    def apply(self, x: np.ndarray):
        n_channels, n_timesteps, n_features = x.shape
        if self.freq_shift_range is None:
            self.freq_shift_range = int(n_features * 0.08)
        shift_len = np.random.randint(1, self.freq_shift_range, 1)[0]
        if self.direction is None:
            direction = np.random.choice(['up', 'down'], 1)[0]
        else:
            direction = self.direction
        new_spec = x.copy()
        if direction == 'up':
            new_spec = np.pad(new_spec, ((0, 0), (0, 0), (shift_len, 0)), mode=self.mode)[:, :, 0: n_features]
        else:
            new_spec = np.pad(new_spec, ((0, 0), (0, 0), (0, shift_len)), mode=self.mode)[:, :, shift_len:]
        return new_spec


class TTA:
    """
    This class collect a list of random augmentation for test data
    """
    def __init__(self, repeat: int = 1):
        self.repeat = repeat
        self.transform_list = [
            AdditiveGaussianNoiseNp(always_apply=True),
            MultiplicativeGaussianNoiseNp(always_apply=True),
            RandomCutoutHoleNp(always_apply=True, n_max_holes=12, max_h_size=6, max_w_size=6),
            RandomRotateNp(always_apply=True),
        ]
        self.n_tta = self.repeat * len(self.transform_list) + 1

    def __call__(self, x: np.ndarray):
        return self.apply(x)

    def apply(self, x: np.ndarray) -> np.ndarray:
        """
        :param x < np.ndarray (n_channels, n_time_steps, n_features)
        :return: tta:  <np.ndarray(n_ttas, n_channels, n_timesteps, n_features)
        """
        tta = [x]
        for _ in np.arange(self.repeat):
            for transform in self.transform_list:
                tta.append(transform(x))
        tta = np.stack(tta, axis=0)
        return tta


