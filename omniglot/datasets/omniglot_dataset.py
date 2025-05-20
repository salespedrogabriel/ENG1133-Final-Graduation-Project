import random
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

class OmniglotTrain(Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform
        np.random.seed(0)

    def __len__(self):
        return 21000000

    def __getitem__(self, index):
        image1 = random.choice(self.dataset.imgs)
        if index % 2 == 1:
            while True:
                image2 = random.choice(self.dataset.imgs)
                if image1[1] == image2[1]:
                    label = 1.0
                    break
        else:
            while True:
                image2 = random.choice(self.dataset.imgs)
                if image1[1] != image2[1]:
                    label = 0.0
                    break

        image1 = Image.open(image1[0]).convert("L")
        image2 = Image.open(image2[0]).convert("L")
        if self.transform:
            image1 = self.transform(image1)
            image2 = self.transform(image2)

        return image1, image2, np.array([label], dtype=np.float32)

class OmniglotTest(Dataset):
    def __init__(self, dataset, transform=None, times=200, way=20):
        self.dataset = dataset
        self.transform = transform
        self.times = times
        self.way = way
        np.random.seed(1)

    def __len__(self):
        return self.times * self.way

    def __getitem__(self, index):
        idx = index % self.way
        if idx == 0:
            self.img1 = random.choice(self.dataset.imgs)
            while True:
                img2 = random.choice(self.dataset.imgs)
                if self.img1[1] == img2[1]:
                    break
        else:
            while True:
                img2 = random.choice(self.dataset.imgs)
                if self.img1[1] != img2[1]:
                    break

        img1 = Image.open(self.img1[0]).convert("L")
        img2 = Image.open(img2[0]).convert("L")
        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
        return img1, img2
