import numpy as np
import pandas as pd
import os

DATASET = 'kickstarter'


def read_allSamples():
    samples = ([], [], [])
    data = pd.read_csv(os.path.abspath(f"../../datasets/{DATASET}/{DATASET}.csv"))
    record_num = data.shape[0]
    for record_id in range(record_num):
        record = data.loc[record_id]
        a_sample = Sample(record)
        samples[a_sample.dataset_type].append(a_sample)
    return samples


class Sample:
    def __init__(self, record):
        self.sample_id = str(record['sample_id'])
        self.state = record['state']
        self.dataset_type = record['dataset_type']
        self.sentence_num = record['sentence_num']
        self.frame_num = min(record['frame_num'], 300)
        self.features_dir_path = os.path.abspath(f"../../datasets/{DATASET}/features/{self.sample_id}")

    def text_features(self):
        features_path = f"{self.features_dir_path}/text_clip.npz"
        data = np.load(features_path)
        features = data["features"]
        return features

    def frame_features(self):
        features_path = f"{self.features_dir_path}/frame_clip.npz"
        data = np.load(features_path)
        features = data["features"]
        features = features[: self.frame_num]
        return features

