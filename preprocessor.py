import torch
from torchvision.datasets import CIFAR10
from decouple import config
from torch.utils.data import DataLoader, TensorDataset
from torchvision.datasets import CIFAR10
from sklearn.model_selection import train_test_split



def load_data_camelyon(preprocess, train, val, test, device):

    print(f'trainset size {len(train)}')
    print(f'validation_set size {len(val)}')
    print(f'test size {len(test)}')
    
    batch_size = int(config('batch_size'))
    train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True)
    validation_loader = torch.utils.data.DataLoader(val, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test, batch_size=batch_size, shuffle=False)

    return train_loader, validation_loader, test_loader


def load_data_cifar(preprocess, train, test, device):
    # train = CIFAR10(root, download=True, train=True)
    train_indices, validation_indices = train_test_split(range(len(train)), test_size=0.2, random_state=42)
    train_set = torch.utils.data.Subset(train, train_indices)
    validation_set = torch.utils.data.Subset(train, validation_indices)
    train_set.dataset.transform = preprocess
    validation_set.dataset.transform = preprocess
    # test = CIFAR10(root, download=True, train=False, transform=preprocess)

    # text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}") for c in test.classes]).to(device)
    print(f'trainset size {len(train_set)}')
    print(f'validation_set size {len(validation_set)}')
    print(f'test size {len(test)}')
    
    batch_size = int(config('batch_size'))
    trainloaders = [torch.utils.data.DataLoader(train_set, batch_size=int(config('batch_size')), shuffle=True) for i in range(int(config('opt')))]
    # train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    validation_loader = torch.utils.data.DataLoader(validation_set, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)
    test_loader = DataLoader(test, batch_size=batch_size, shuffle=False)

    return trainloaders, validation_loader, test_loader


def load_data_imagenet( train_set, val_dataset, test_dataset, device):
    torch.manual_seed(42)

    print(f'trainset size {len(train_set)}')
    print(f'validation_set size {len(val_dataset)}')
    print(f'test size {len(test_dataset)}')
    
    # train_loader = DataLoader(train_set, batch_size=int(config('batch_size')), shuffle=True, num_workers=8, pin_memory=True)
    trainloaders = [torch.utils.data.DataLoader(train_set, batch_size=int(config('batch_size')), shuffle=True) for i in range(int(config('opt')))]
    val_loader = DataLoader(val_dataset, batch_size=int(config('batch_size')), shuffle=False, num_workers=8, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=int(config('batch_size')), shuffle=False, num_workers=8, pin_memory=True)

    return trainloaders, val_loader, test_loader
