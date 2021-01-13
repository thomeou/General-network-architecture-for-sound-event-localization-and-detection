"""
This module includes labels mapping for different dataset and different setting
"""


class SELD2020Config:
    """
    Data configs for SELD 2020 dataset.
    """
    def __init__(self):
        self.event_labels = ['alarm', 'baby', 'crash', 'dog', 'engine',
                             'female_scream', 'female_speech', 'fire', 'footsteps',
                             'knock', 'male_scream', 'male_speech', 'phone', 'piano']
        self.n_classes = len(self.event_labels)
        self.lb_to_ix = {lb: i for i, lb in enumerate(self.event_labels)}
        self.ix_to_lb = {i: lb for i, lb in enumerate(self.event_labels)}