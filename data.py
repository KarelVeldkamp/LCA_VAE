from torch.utils.data import Dataset
import pandas as pd
import torch


class CSVDataset(Dataset):
    """
    Torch dataset for item response data in numpy array
    """
    def __init__(self, path, device='cpu'):
        """
        initialize
        :param file_name: path to csv that contains NXI matrix of responses from N people to I items
        """
        # Read csv and ignore rownames

        X = pd.read_csv(path, index_col=0).values
        X -= 1
        self.x_train = torch.tensor(X, dtype=torch.float32)

    def __len__(self):
        return len(self.x_train)

    def __getitem__(self, idx):
        return self.x_train[idx]


class MNISTDataset(Dataset):
    """
    Torch dataset for item response data in numpy array
    """
    def __init__(self, path, device='cpu'):
        """
        initialize
        :param file_name: path to csv that contains NXI matrix of responses from N people to I items
        """
        # Read csv and ignore rownames

        X = pd.read_csv(path, index_col=0).iloc[:,1:].values
        self.x_train = torch.tensor(X, dtype=torch.float32)

    def __len__(self):
        return len(self.x_train)

    def __getitem__(self, idx):
        return self.x_train[idx]