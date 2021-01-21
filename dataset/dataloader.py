"""
Module for dataloader
"""
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

from dataset.database import SedDoaDatabase


class SedDoaChunkDataset(Dataset):
    """
    Chunk dataset for SED or DOA task. For training and chunk evaluation.
    """
    def __init__(self, db_data, chunk_len, transform=None, is_mixup: bool = False):
        self.features = db_data['features']
        self.sed_targets = db_data['sed_targets']
        self.doa_targets = db_data['doa_targets']
        self.chunk_idxes = db_data['chunk_idxes']
        self.filename_list = db_data['filename_list']
        self.chunk_len = chunk_len
        self.transform = transform
        self.is_mixup = is_mixup
        self.n_samples = len(self.chunk_idxes)

    def __len__(self):
        """
        Total of training samples.
        """
        return len(self.chunk_idxes)

    def __getitem__(self, index):
        """
        Generate one sample of data
        """
        # Select sample
        chunk_idx = self.chunk_idxes[index]

        # get filename
        filename = self.filename_list[index]

        # Load data and get label
        X = self.features[:, chunk_idx: chunk_idx + self.chunk_len, :]  # (n_channels, n_timesteps, n_mels)
        sed_labels = self.sed_targets[chunk_idx: chunk_idx + self.chunk_len]  # (n_timesteps, n_classes)
        doa_labels = self.doa_targets[chunk_idx: chunk_idx + self.chunk_len]  # (n_timesteps, x*n_classes)

        # Mixup mainly for SED
        if self.is_mixup:
            a1 = np.random.beta(0.5, 0.5)
            if np.random.rand() < 0.8 and np.abs(a1 - 0.5) > 0.2:
                random_index = np.random.randint(0, self.n_samples, 1)[0]
                random_chunk_idx = self.chunk_idxes[random_index]
                X_1 = self.features[:, random_chunk_idx: random_chunk_idx + self.chunk_len, :]
                sed_labels_1 = self.sed_targets[random_chunk_idx: random_chunk_idx + self.chunk_len]
                doa_labels_1 = self.doa_targets[random_chunk_idx: random_chunk_idx + self.chunk_len]
                X = a1 * X + (1 - a1) * X_1
                sed_labels = a1 * sed_labels + (1 - a1) * sed_labels_1
                doa_labels = a1 * doa_labels + (1 - a1) * doa_labels_1

        if self.transform is not None:
            X = self.transform(X)

        return X, sed_labels, doa_labels, filename


class SeldChunkDataset(Dataset):
    """
    Chunk dataset for SELD task
    """
    pass


if __name__ == '__main__':
    # test dataloader
    db = SedDoaDatabase()
    data_db = db.get_split(split='val')

    # create train dataset
    dataset = SedDoaChunkDataset(db_data=data_db, chunk_len=db.chunk_len)
    print('Number of training samples: {}'.format(len(dataset)))

    # load one sample
    index = np.random.randint(len(dataset))
    sample = dataset[index]
    for item in sample[:-1]:
        print(item.shape)
    print(sample[-1])

    # test data generator
    batch_size = 8
    dataloader = torch.utils.data.DataLoader(dataset=dataset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             num_workers=4)
    print('Number of batches: {}'.format(len(dataloader)))  # steps_per_epoch
    for train_iter, (X, sed_labels, doa_labels, filenames) in enumerate(dataloader):
        if train_iter == 0:
            print(X.dtype)
            print(X.shape)
            print(sed_labels.dtype)
            print(sed_labels.shape)
            print(doa_labels.dtype)
            print(doa_labels.shape)
            print(type(filenames))
            print(filenames)
            break