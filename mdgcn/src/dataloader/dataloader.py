import random
import torch
import numpy as np
from src.dataloader import fsample


TRAIN = 0
VAL = 1
TEST = 2


class Dataloader:
    def __init__(self, dataset_type, batch_size, device):
        assert dataset_type in [TRAIN, VAL, TEST]
        self.dataset_type = dataset_type
        self.device = device
        self.batch_size = batch_size
        self.samples = fsample.read_allSamples()[self.dataset_type]
        self.sample_num = len(self.samples)
        self.index = 0
        self.iter_num = -1

    def __iter__(self):
        if self.dataset_type == TRAIN:
            random.shuffle(self.samples)
        self.sample_blocks = []
        i = 0
        while True:
            if i >= self.sample_num:
                break
            self.sample_blocks.append(self.samples[i: i + self.batch_size])
            i += self.batch_size
        self.index = 0
        self.iter_num = len(self.sample_blocks)
        return self

    # ([[batch_text_features, batch_text_mask], [batch_frame_features, batch_frame_mask]], batch_labels)
    def __next__(self):
        if self.index >= self.iter_num:
            raise StopIteration
        batch_samples = self.sample_blocks[self.index]
        batch_text_data = self.get_tf_data(batch_samples, frame=False)
        batch_frame_data = self.get_tf_data(batch_samples, frame=True)
        batch_inputs = [batch_text_data, batch_frame_data]
        batch_labels = self.get_label_data(batch_samples)
        self.index += 1
        return batch_inputs, batch_labels

    def __len__(self):
        return self.sample_num

    # text features or frame features
    def get_tf_data(self, batch_samples, frame, vec_len=512):
        batch_size = len(batch_samples)
        max_seq_len = 0
        for each in batch_samples:
            max_seq_len = max(max_seq_len, each.frame_num if frame else each.sentence_num)
        # get features and masks
        batch_features = torch.zeros((batch_size, max_seq_len, vec_len))
        batch_masks = torch.zeros((batch_size, max_seq_len), dtype=torch.bool)
        for i in range(batch_size):
            sample = batch_samples[i]
            src_data = sample.frame_features() if frame else sample.text_features()
            effective_len, _ = src_data.shape
            padding_len = max_seq_len - effective_len
            # feature
            feature_cell = np.concatenate([src_data, np.zeros((padding_len, vec_len))], axis=0)
            feature_cell = torch.tensor(feature_cell)
            batch_features[i] = feature_cell
            # mask
            mask_cell = np.zeros((max_seq_len, ), dtype=np.bool8)
            mask_cell[effective_len:] = True
            batch_masks[i] = torch.tensor(mask_cell)
        batch_features = batch_features.to(self.device)
        batch_masks = batch_masks.to(self.device)
        batch_data = [batch_features, batch_masks]
        return batch_data

    def get_label_data(self, batch_samples):
        batch_size = len(batch_samples)
        batch_data = torch.zeros(batch_size, dtype=torch.long)
        for i in range(batch_size):
            sample = batch_samples[i]
            batch_data[i] = torch.tensor(sample.state)
        batch_data = batch_data.to(self.device)
        return batch_data

