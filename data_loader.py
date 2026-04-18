from PIL import Image
import matplotlib.pyplot as plt
import numpy as np


class idxLoader:
    def __init__(self):
        # train img, train lbl, test img, test lbl
        self.path1 = 'data/train-images-idx3-ubyte'
        self.path2 = 'data/train-labels-idx1-ubyte'
        self.path3 = 'data/t10k-images-idx3-ubyte'
        self.path4 = 'data/t10k-labels-idx1-ubyte'

    def train_img_loader(self):
        with open(self.path1, 'rb') as f:
            data = np.frombuffer(f.read(), dtype=np.uint8, offset=16)
        images = data.reshape(-1, 28*28)
        images = images.astype(np.float32) / 255.0
        return images
    
    def train_lbl_loader(self):
        with open(self.path2, 'rb') as f:
            data = np.frombuffer(f.read(), dtype=np.uint8, offset=8)
        labels = data.astype(np.int64)
        return labels

    def test_img_loader(self):
        with open(self.path3, 'rb') as f:
            data = np.frombuffer(f.read(), dtype=np.uint8, offset=16)
        images = data.reshape(-1, 28*28)
        images = images.astype(np.float32) / 255.0
        return images

    def test_lbl_loader(self):
        with open(self.path4, 'rb') as f:
            data = np.frombuffer(f.read(), dtype=np.uint8, offset=8)
        labels = data.astype(np.int64)
        return labels

    def visualize(self, img):
        img = img.reshape(28, 28)
        plt.imshow(img, cmap='gray')
        plt.axis('off')
        plt.show()