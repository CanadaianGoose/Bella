import open_clip
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import Dataset
import torch
from PIL import Image
import os


class TrainDataset(Dataset):
    def __init__(self, data_folder, transform=None):
        self.image_folder = ImageFolder(root=data_folder, transform=transform)

    def __len__(self):
        return len(self.image_folder)

    def __getitem__(self, idx):
        return self.image_folder[idx]




class ValDataset(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.images, self.labels = self.load_data()

    def load_data(self):
        images = []
        labels = []
        labels_file = os.path.join(self.root, 'ILSVRC2010_validation_ground_truth.txt')
        with open(labels_file, 'r') as file:
            labels = [int(line.strip())-1 for line in file]

        for filename in os.listdir(self.root): 
            if filename.endswith(".JPEG"):
                image_path = os.path.join(self.root, filename)
                images.append(image_path)

        return images, labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = self.images[idx]
        label = self.labels[idx]
        with open(image_path, 'rb') as f:
            image = Image.open(f)
            if self.transform:
                image = self.transform(image)
        return image, label



