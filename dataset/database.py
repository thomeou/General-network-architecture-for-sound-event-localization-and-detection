"""
This module consists database class to handle different different data split, loading data into memory, dividing long
audio file into segments and tokenize these segments
Note: 32bit is sufficient for DL models.
Note on terminology on audio length: frames -> segments -> chunk/clip -> file
"""
import json
import logging
import os
from typing import List

import h5py
import numpy as np
import pandas as pd


class SedDoaDatabase:
    """
    Class to handle different extracted features for SED or DOA separately.
    TODO: add background class
    """
    def __init__(self,
                 feature_root_dir: str = '/media/tho_nguyen/disk2/new_seld/dcase2020/features/'
                                         'logmel_norm/24000fs_1024nfft_300nhop_128nmels_Falsestd',
                 gt_meta_root_dir: str = '/media/tho_nguyen/disk1/audio_datasets/dcase2020/task3',
                 audio_format: str = 'foa', n_classes: int = 14, fs: int = 24000, n_fft: int = 1024, hop_len: int = 300,
                 label_rate: float = 10, train_chunk_len_s: float = 4.0, train_chunk_hop_len_s: float = 0.5,
                 test_chunk_len_s: float = 4.0, test_chunk_hop_len_s: float = 2.0, scaler_type: str = 'scalar'):
        """
        :param feature_root_dir: Feature directory. can be SED or DOA feature.
        The data are organized in the following format:
            |__feature_root_dir/
                |__foa_dev/
                |__foa_eval/
                |__mic_dev/
                |__mic_eval/
                |__foa_feature_scaler.h5
                |__mic_feature_scaler.h5
        :param gt_meta_root_dir: Directory that contains groundtruth meta data.
        The data are orgamized in the following format:
            |__gt_meta_dir/
                |__/metadata_dev/
                |__/metadata_eval/
                |__metadata_eval_info.csv
        """
        self.feature_root_dir = feature_root_dir
        self.gt_meta_root_dir = gt_meta_root_dir
        self.audio_format = audio_format
        self.n_classes = n_classes
        self.fs = fs
        self.n_fft = n_fft
        self.hop_len = hop_len
        self.label_rate = label_rate
        self.train_chunk_len = self.second2frame(train_chunk_len_s)
        self.train_chunk_hop_len = self.second2frame(train_chunk_hop_len_s)
        self.test_chunk_len = self.second2frame(test_chunk_len_s)
        self.test_chunk_hop_len = self.second2frame(test_chunk_hop_len_s)
        self.scaler_type = scaler_type

        assert audio_format in ['foa', 'mic'], 'Incorrect value for audio format {}'.format(audio_format)
        assert os.path.isdir(os.path.join(self.feature_root_dir, self.audio_format + '_dev')), \
            '"dev" folder is not found'

        self.chunk_len = None
        self.chunk_hop_len = None
        self.n_frames = int(np.floor((self.fs * 60 - (self.n_fft - self.hop_len)) / self.hop_len)) + 2  #+ 2 because of padding
        self.feature_rate = self.fs/self.hop_len  # Frame rate per second
        self.label_upsample_ratio = int(self.feature_rate / self.label_rate)


        self.mean, self.std = self.load_scaler()

        logger = logging.getLogger('lightning')
        logger.info('Load feature database from {}.'.format(self.feature_root_dir))
        logger.info('train_chunk_len = {}, train_chunk_hop_len = {}'.format(
            self.train_chunk_len, self.train_chunk_hop_len))
        logger.info('test_chunk_len = {}, test_chunk_hop_len = {}'.format(
            self.test_chunk_len, self.test_chunk_hop_len))

    def get_split(self, split: str, split_meta_dir: str = 'meta/original', doa_format: str = 'xyz'):
        """
        Function to load all data of a split into memory, divide long audio clip/file into smaller chunks, and assign
        labels for clips and chunks. List of SED labels:

        :param split: Split of data, choices:
            'train', 'val', 'test', 'eval': load chunk of data
        :param split_meta_dir: Directory where meta of split is stored.
        :param doa_format: Choices are 'xyz' or 'polar'.
        :return:
        """
        assert doa_format in ['xyz', 'polar'], 'Incorrect value for doa format {}'.format(doa_format)
        # Get feature dir, filename list, and gt_meta_dir
        if split == 'eval':
            split_feature_dir = os.path.join(self.feature_root_dir, self.audio_format + '_eval')
            csv_filename = os.path.join(os.path.split(split_meta_dir)[0], 'eval.csv')
            gt_meta_dir = os.path.join(self.gt_meta_root_dir, 'metadata_eval')
        else:
            split_feature_dir = os.path.join(self.feature_root_dir, self.audio_format + '_dev')
            csv_filename = os.path.join(split_meta_dir, split + '.csv')
            gt_meta_dir = os.path.join(self.gt_meta_root_dir, 'metadata_dev')
        meta_df = pd.read_csv(csv_filename)
        split_filenames = meta_df['filename'].tolist()
        # Get chunk len and chunk hop len
        if split in ['train', 'trainval', 'val']:
            self.chunk_len = self.train_chunk_len
            self.chunk_hop_len = self.train_chunk_hop_len
        elif split in ['test', 'eval']:
            self.chunk_len = self.test_chunk_len
            self.chunk_hop_len = self.test_chunk_hop_len
        else:
            raise NotImplementedError('chunk len is not assigned for split {}'.format(split))

        # Load and crop data
        features, sed_targets, doa_targets, chunk_idxes, filename_list = self.load_chunk_data(
            split_filenames=split_filenames, split_feature_dir=split_feature_dir, gt_meta_dir=gt_meta_dir,
            doa_format=doa_format, split=split)
        # pack data
        db_data = {
            'features': features,
            'sed_targets': sed_targets,
            'doa_targets': doa_targets,
            'chunk_idxes': chunk_idxes,
            'filename_list': filename_list
        }

        return db_data

    def second2frame(self, second):
        """
        Convert seconds to frame unit.
        """
        sample = int(second * self.fs)
        frame = int(round(sample/self.hop_len))
        return frame

    def load_scaler(self):
        scaler_fn = os.path.join(self.feature_root_dir, self.audio_format + '_feature_scaler.h5')
        if self.scaler_type == 'vector':
            with h5py.File(scaler_fn, 'r') as hf:
                mean = hf['mean'][:]
                std = hf['std'][:]
        elif self.scaler_type == 'scalar':
            with h5py.File(scaler_fn, 'r') as hf:
                mean = hf['scalar_mean'][:]
                std = hf['scalar_std'][:]
        else:
            mean = 0
            std = 1
        return mean, std

    def load_chunk_data(self, split_filenames: List, split_feature_dir: str, gt_meta_dir: str,
                        doa_format: str = 'xyz', split: str ='train'):
        """
        Load feature, crop data and assign labels.
        :param split_filenames: List of filename in the split.
        :param split_feature_dir: Feature directory of the split
        :param gt_meta_dir: Ground truth meta directory of the split.
        :param doa_format: Choices are 'xyz' or 'polar'.
        :param split: Name of split, can be 'train', 'trainval', 'val', 'test', 'eval'.
        :return: features, targets, chunk_idxes, filename_list
        """
        pointer = 0
        features_list = []
        filename_list = []
        sed_targets_list = []
        doa_targets_list = []
        idxes_list = []
        for filename in split_filenames:
            feature_fn = os.path.join(split_feature_dir, filename + '.h5')
            # Load feature
            with h5py.File(feature_fn, 'r') as hf:
                feature = hf['feature'][:]
            # Normalize feature
            feature = (feature - self.mean) / self.std
            n_frames = feature.shape[1]
            # Load gt info from metadata
            gt_meta_fn = os.path.join(gt_meta_dir, filename + '.csv')
            df = pd.read_csv(gt_meta_fn, header=None,
                             names=['frame_number', 'sound_class_idx', 'track_number', 'azimuth', 'elevation'])
            frame_number = df['frame_number'].values
            sound_class_idx = df['sound_class_idx'].values
            track_number = df['track_number'].values
            azimuth = df['azimuth'].values
            elevation = df['elevation'].values
            # Generate target data
            sed_target = np.zeros((n_frames, self.n_classes), dtype=np.float32)
            azi_target = np.zeros((n_frames, self.n_classes), dtype=np.float32)
            ele_target = np.zeros((n_frames, self.n_classes), dtype=np.float32)
            nsources_target = np.zeros((n_frames, 3), dtype=np.float32)
            count_sources_target = np.zeros((n_frames,), dtype=np.float32)
            for itrack in np.arange(5):
                track_idx = track_number == itrack
                frame_number_1 = frame_number[track_idx]
                sound_class_idx_1 = sound_class_idx[track_idx]
                azimuth_1 = azimuth[track_idx]
                elevation_1 = elevation[track_idx]
                for idx, iframe in enumerate(frame_number_1):
                    start_idx = int(iframe * self.label_upsample_ratio - self.label_upsample_ratio//2)
                    start_idx = np.max((0, start_idx))
                    end_idx = int(start_idx + self.label_upsample_ratio)
                    end_idx = np.min((end_idx, n_frames))
                    class_idx = int(sound_class_idx_1[idx])
                    sed_target[start_idx:end_idx, class_idx] = 1.0
                    azi_target[start_idx:end_idx, class_idx] = azimuth_1[idx] * np.pi / 180.0  # Radian unit
                    ele_target[start_idx:end_idx, class_idx] = elevation_1[idx] * np.pi / 180.0  # Radian unit
                    count_sources_target[start_idx:end_idx] += 1
            # Convert nsources to one-hot encoding
            for i in np.arange(3):
                idx = count_sources_target == i
                nsources_target[idx, i] = 1.0
            # Doa target
            if doa_format == 'polar':
                doa_target = np.concatenate((azi_target, ele_target), axis=-1)
            elif doa_format == 'xyz':
                x = np.cos(azi_target) * np.cos(ele_target)
                y = np.sin(azi_target) * np.cos(ele_target)
                z = np.sin(ele_target)
                doa_target = np.concatenate((x, y, z), axis=-1)
            # Get segment indices
            n_crop_frames = n_frames
            assert self.chunk_len <= n_crop_frames, 'Number of cropped frame is less than chunk len'
            idxes = np.arange(pointer, pointer + n_crop_frames - self.chunk_len + 1, self.chunk_hop_len).tolist()
            # Include the leftover of the cropped data
            if (n_crop_frames - self.chunk_len) % self.chunk_hop_len != 0:
                idxes.append(pointer + n_crop_frames - self.chunk_len)
            pointer += n_crop_frames
            # Append data
            features_list.append(feature)
            filename_list.extend([filename] * len(idxes))
            sed_targets_list.append(sed_target)
            doa_targets_list.append(doa_target)
            idxes_list.append(idxes)

        if len(features_list) > 0:
            features = np.concatenate(features_list, axis=1)
            sed_targets = np.concatenate(sed_targets_list, axis=0)
            doa_targets = np.concatenate(doa_targets_list, axis=0)
            chunk_idxes = np.concatenate(idxes_list, axis=0)
            return features, sed_targets, doa_targets, chunk_idxes, filename_list
        else:
            return None, None, None, None

class SeldDatabase():
    """
    Database class to handle two input streams, one for SED, one for DOA. Use this database to train alignment module
    """
    pass


if __name__ == '__main__':
    tp_db = SedDoaDatabase()
    db_data = tp_db.get_split(split='val', split_meta_dir='meta/original')
    print(db_data['features'].shape)
    print(db_data['sed_targets'].shape)
    print(len(db_data['chunk_idxes']))
    print(len(db_data['filename_list']))
