from torch.utils.data import Dataset
import os
import cv2 as cv
import numpy as np
from torchvision import transforms


class Tumor(Dataset):
    def __init__(self, path, folders):
        self.path = path
        self.folders = folders
        self.yes_path = self.path + "\\yes"
        self.no_path = self.path + "\\no"
        self.yes = os.listdir(self.yes_path)
        self.no = os.listdir(self.no_path)
        self.transform = transforms.Compose([transforms.ToTensor()])

    def __len__(self):
        return len(self.yes) + len(self.no)

    def __getitem__(self, item):
        images = []
        labels = []
        for i in self.yes:
            images.append(os.path.join(self.yes_path, i))
            labels.append(1)

        for i in self.no:
            images.append(os.path.join(self.no_path, i))
            labels.append(0)

        img = images[item]
        label = labels[item]

        mat = cv.imread(img, 0)
        mat = cv.resize(mat, (100, 100))
        im_pil = np.array(mat)
        img = self.transform(im_pil)
        return img, label
