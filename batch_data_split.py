from dataset import Tumor
from torch.utils.data import DataLoader
from torch.utils.data import random_split
import utils

data = Tumor("D:\\Datasets\\Computer Vision\\Brain Tumor", ['yes', 'no'])

train_len = int(0.9 * len(data))
test_len = len(data) - train_len

train, test = random_split(data, [train_len, test_len])

train_data = DataLoader(
    train, batch_size=utils.batch_size, shuffle=True
)

test_data = DataLoader(
    test, batch_size=utils.batch_size
)
