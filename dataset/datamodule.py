import logging

import pytorch_lightning as pl
from torch.utils.data import DataLoader

from dataset.dataloader import SedDoaChunkDataset
from utilities.transforms import CompositeCutout


class SedDoaDataModule(pl.LightningDataModule):
    """
    DataModule that group train and validation data for SED or DOA task loader under on hood.
    """
    def __init__(self, feature_db, split_meta_dir: str = '/dataset/meta/original/', train_batch_size: int = 32,
                 val_batch_size: int = 32, mode: str = 'crossval'):
        super().__init__()
        self.feature_db = feature_db
        self.split_meta_dir = split_meta_dir
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.test_batch_size = None
        self.lit_logger = logging.getLogger('lightning')
        self.lit_logger.info('Create DataModule using tran val split at {}.'.format(split_meta_dir))
        if mode == 'crossval':
            self.train_split = 'train'
            self.val_split = 'val'
            self.test_split = 'test'
        elif mode == 'eval':
            self.train_split = 'trainval'
            self.val_split = 'test'
            self.test_split = 'eval'
        else:
            raise NotImplementedError('Mode {} is not implemented!'.format(mode))

        # Data augmentation
        self.train_transform = CompositeCutout(image_aspect_ratio=self.feature_db.train_chunk_len/128)  # 128 n_mels

    def setup(self, stage: str = None):
        """
        :param stage: can be 'fit', 'test.
        """
        # Get train and val data during training
        if stage == 'fit':  # to use clip for validation
            train_db = self.feature_db.get_split(split=self.train_split, split_meta_dir=self.split_meta_dir)
            self.train_dataset = SedDoaChunkDataset(db_data=train_db, chunk_len=self.feature_db.train_chunk_len,
                                                    transform=self.train_transform, is_mixup=True)
            val_db = self.feature_db.get_split(split=self.val_split, split_meta_dir=self.split_meta_dir)
            self.val_dataset = SedDoaChunkDataset(db_data=val_db, chunk_len=self.feature_db.train_chunk_len)
        elif stage == 'test':
            test_db = self.feature_db.get_split(split=self.test_split, split_meta_dir=self.split_meta_dir)
            self.test_dataset = SedDoaChunkDataset(db_data=test_db, chunk_len=self.feature_db.test_chunk_len)
            self.test_batch_size = test_db['test_batch_size']
            self.lit_logger.info('In datamodule: test batch size = {}'.format(self.test_batch_size))
        else:
            raise NotImplementedError('stage {} is not implemented for datamodule'.format(stage))

    def train_dataloader(self):
        return DataLoader(dataset=self.train_dataset,
                          batch_size=self.train_batch_size,
                          shuffle=True,
                          pin_memory=True,
                          num_workers=4)

    def val_dataloader(self):
        return DataLoader(dataset=self.val_dataset,
                          batch_size=self.val_batch_size,
                          shuffle=False,
                          pin_memory=True,
                          num_workers=4)

    def test_dataloader(self):
        return DataLoader(dataset=self.test_dataset,
                          batch_size=self.test_batch_size,
                          shuffle=False,
                          pin_memory=True,
                          num_workers=4)


class SeldDataModule(pl.LightningDataModule):
    """
    DataModule that group train and validation data for SELD task loader under on hood.
    """