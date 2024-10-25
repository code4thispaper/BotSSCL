import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

class TwitterAccountDatasetAug1(Dataset):
    def __init__(self, data, target, columns=None):
        self.data = np.array(data)
        self.target = np.array(target)
        self.columns = columns

    def __getitem__(self, index):
        # the dataset must return a pair of samples: the anchor and a random one from the
        # dataset that will be used to corrupt the anchor
        # Fit 99 to predit 1 row (in 100 samples)
        # Put NaN in 50% of the data (for example)
        # Put NaN then Predict
        # 2074, 128
        # The row is the anchor
        random_idx = np.random.randint(0, len(self))
        random_sample = torch.tensor(self.data[random_idx], dtype=torch.float)
        sample = torch.tensor(self.data[index], dtype=torch.float)

        return sample, random_sample

    def __len__(self):
        return len(self.data)

    def to_dataframe(self):
        return pd.DataFrame(self.data, columns=self.columns)

    @property
    def shape(self):
        return self.data.shape


class TwitterAccountDatasetAug2(Dataset):
    def __init__(self, data, target, imputed_data, columns=None):
        self.data = np.array(data)
        self.target = np.array(target)
        self.imputed_data = np.array(imputed_data)
        self.columns = columns

    def __getitem__(self, index):
        # the dataset must return a pair of samples: the anchor and a random one from the
        # dataset that will be used to corrupt the anchor
        # Fit 99 to predit 1 row (in 100 samples)
        # Put NaN in 50% of the data (for example)
        # Put NaN then Predict
        # 2074, 128
        # The row is the anchor
        # Aug 2 -> linear transformation of main sample (add another layer)
        random_idx = np.random.randint(0, len(self))
        random_sample = torch.tensor(self.imputed_data[random_idx], dtype=torch.float)
        sample = torch.tensor(self.data[index], dtype=torch.float)

        return sample, random_sample

    def __len__(self):
        return len(self.data)

    def to_dataframe(self):
        return pd.DataFrame(self.data, columns=self.columns)

    @property
    def shape(self):
        return self.data.shape

class TwitterAccountDatasetAug2_1(Dataset):
    def __init__(self, data, target, imputed_data_1, imputed_data_2, columns=None):
        self.data = np.array(data)
        self.target = np.array(target)
        self.imputed_data_view_1 = np.array(imputed_data_1)
        self.imputed_data_view_2 = np.array(imputed_data_2)
        self.columns = columns

    def __getitem__(self, index):
        # the dataset must return a pair of samples: the anchor and a random one from the
        # dataset that will be used to corrupt the anchor
        # Fit 99 to predit 1 row (in 100 samples)
        # Put NaN in 50% of the data (for example)
        # Put NaN then Predict
        # 2074, 128
        # The row is the anchor
        # Aug 2 -> linear transformation of main sample (add another layer)
        random_idx = np.random.randint(0, len(self))
        # Create two imputed files (only one more to go)
        imputed_random_sample = torch.tensor(self.imputed_data_view_2[random_idx], dtype=torch.float)
        imputed_sample = torch.tensor(self.imputed_data_view_1[index], dtype=torch.float)

        return imputed_sample, imputed_random_sample

    def __len__(self):
        return len(self.data)

    def to_dataframe(self):
        return pd.DataFrame(self.data, columns=self.columns)

    @property
    def shape(self):
        return self.data.shape

class TwitterAccountDatasetAug3(Dataset):
    def __init__(self, data, target, columns=None):
        self.data = np.array(data)
        self.target = np.array(target)
        self.columns = columns

    def __getitem__(self, index):
        # the dataset must return a pair of samples: the anchor and a random one from the
        # dataset that will be used to corrupt the anchor
        # Fit 99 to predit 1 row (in 100 samples)
        # Put NaN in 50% of the data (for example)
        # Put NaN then Predict
        # 2074, 128
        # The row is the anchor
        random_idx = np.random.randint(0, len(self))
        random_sample = torch.tensor(self.data[random_idx], dtype=torch.float)
        sample = torch.tensor(self.data[index], dtype=torch.float)
        linear = torch.nn.Linear(len(self.data[index]), len(self.data[index]))
        sample_transformed = linear(sample)

        return sample, sample_transformed

    def __len__(self):
        return len(self.data)

    def to_dataframe(self):
        return pd.DataFrame(self.data, columns=self.columns)

    @property
    def shape(self):
        return self.data.shape

class TwitterAccountDatasetAug3_1(Dataset):
    def __init__(self, data, target, columns=None):
        self.data = np.array(data)
        self.target = np.array(target)
        self.columns = columns

    def __getitem__(self, index):
        # the dataset must return a pair of samples: the anchor and a random one from the
        # dataset that will be used to corrupt the anchor
        # Fit 99 to predit 1 row (in 100 samples)
        # Put NaN in 50% of the data (for example)
        # Put NaN then Predict
        # 2074, 128
        # The row is the anchor
        random_idx = np.random.randint(0, len(self))
        random_sample = torch.tensor(self.data[random_idx], dtype=torch.float)
        sample = torch.tensor(self.data[index], dtype=torch.float)
        # Check if they create different vectors
        linear1 = torch.nn.Linear(len(self.data[index]), len(self.data[index]))
        linear2 = torch.nn.Linear(len(self.data[index]), len(self.data[index]))
        sample_transformed_1 = linear1(sample)
        sample_transformed_2 = linear2(sample)

        return sample_transformed_1, sample_transformed_2

    def __len__(self):
        return len(self.data)

    def to_dataframe(self):
        return pd.DataFrame(self.data, columns=self.columns)

    @property
    def shape(self):
        return self.data.shape
