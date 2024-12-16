import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision.io import read_image
from torchvision.transforms import Compose, Resize, Normalize


class WikiArtImage:
    def __init__(self, imgdir, label, filename):
        self.imgdir = imgdir
        self.label = label
        self.filename = filename
        self.image = None
        self.loaded = False

    def get(self):
        if not self.loaded:
            filepath = os.path.join(self.imgdir, self.label, self.filename)
            self.image = read_image(filepath).float() / 255  #normalized to [0, 1]
            self.loaded = True
        return self.image


class WikiArtDataset(Dataset):
    """
    Custom dataset to handle WikiArt image classification tasks.
    """
    def __init__(self, imgdir, transform=None, device="cpu"):
        self.filedict = {}
        self.indices = []
        self.classes = set()
        self.label_counts = {}
        self.labels_str = []
        self.imgdir = imgdir
        self.transform = transform
        self.device = device

        #traversing the directory and collecting image paths
        for root, _, files in os.walk(imgdir):
            label = os.path.basename(root)
            for filename in files:
                self.filedict[filename] = WikiArtImage(imgdir, label, filename)
                self.indices.append(filename)
                self.classes.add(label)
                self.labels_str.append(label)
                self.label_counts[label] = self.label_counts.get(label, 0) + 1

        #sorting classes to maintain consistent indexing
        self.classes = sorted(list(self.classes))
        self.label_to_idx = {label: i for i, label in enumerate(self.classes)}
        self.labels = [self.label_to_idx[label] for label in self.labels_str]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        """
        Retrieve image and label for the given index.
        """
        imgname = self.indices[idx]
        imgobj = self.filedict[imgname]
        image = imgobj.get().to(self.device)
        label = self.label_to_idx[imgobj.label]

        if self.transform:
            image = self.transform(image)

        return image, label


class WikiArtModel(nn.Module):
    """
    Base classification model with options for Bonus A.
    """
    def __init__(self, num_classes=27, bonusA=False):
        super(WikiArtModel, self).__init__()

        if bonusA:
            #modified architecture for Bonus A
            self.conv2d = nn.Conv2d(3, 1, kernel_size=(4, 4), padding=2)
            self.pool = nn.AdaptiveAvgPool2d((50, 50))
            self.flatten = nn.Flatten()
            self.batchnorm1d = nn.BatchNorm1d(50 * 50)
            self.linear1 = nn.Linear(50 * 50, 300)
            self.dropout = nn.Dropout(0.01)
            self.activation = nn.Sigmoid()
            self.linear2 = nn.Linear(300, num_classes)
            self.softmax = nn.LogSoftmax(dim=1)
        else:
            #default architecture
            self.conv2d = nn.Conv2d(3, 1, kernel_size=(4, 4), padding=2)
            self.pool = nn.MaxPool2d((4, 4), padding=2)
            self.flatten = nn.Flatten()
            self.batchnorm1d = nn.BatchNorm1d(105 * 105)
            self.linear1 = nn.Linear(105 * 105, 300)
            self.dropout = nn.Dropout(0.01)
            self.activation = nn.ReLU()
            self.linear2 = nn.Linear(300, num_classes)
            self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.conv2d(x)
        x = self.pool(x)
        x = self.flatten(x)
        x = self.batchnorm1d(x)
        x = self.linear1(x)
        x = self.dropout(x)
        x = self.activation(x)
        x = self.linear2(x)
        return self.softmax(x)


class Autoencoder(nn.Module):
    """
    Autoencoder for Part 2 (used in testtrainauto) (Learning latent representations).
    """
    def __init__(self):
        super(Autoencoder, self).__init__()
        #encoder, compress the input
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )

        #decoder, reconstruct the input
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 3, kernel_size=3, stride=2, padding=1, output_padding=1),
            #sigmoid for pixel values between 0 and 1
            nn.Sigmoid()  
        )

    def forward(self, x):
        latent = self.encoder(x)  
        reconstruction = self.decoder(latent)  
        return latent, reconstruction
