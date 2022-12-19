import numpy as np
import scipy.io as sio
from torch.utils.data import Dataset

class Multi_view_data(Dataset):
    """
    load multi-view data
    """

    def __init__(self, root, loader, view_number):
        """
        :param root: data name and path
        :param train: load training set or test set
        """
        super(Multi_view_data, self).__init__()
        self.root = root
        data_path = self.root + '.mat'

        dataset = sio.loadmat(data_path)

        self.view_number = view_number
        self.X = dict()
        if loader == 'train':
            for v_num in range(view_number):
                self.X[v_num] = (dataset['x' + str(v_num + 1) + '_train'])
            y = dataset['gt_train']
        elif loader == 'val':
            for v_num in range(view_number):
                self.X[v_num] = (dataset['x' + str(v_num + 1) + '_val'])
            y = dataset['gt_val']
        elif loader == 'test':
            for v_num in range(view_number):
                self.X[v_num] = (dataset['x' + str(v_num + 1) + '_test'])
            y = dataset['gt_test']
        tmp = np.zeros(y.shape[0])
        y = np.reshape(y, np.shape(tmp))
        self.y = y

    def __getitem__(self, index):
        data = dict()
        for v_num in range(len(self.X)):
            data[v_num] = (self.X[v_num][index]).astype(np.float32)
        target = self.y[index]
        return data, target

    def __len__(self):
        return len(self.X[0])