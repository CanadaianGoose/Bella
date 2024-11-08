
import pickle
import os
import open_clip
import copy
import random
import ast
import torch
import json
import pandas as pd
import numpy as np
import torch.nn.functional as F
from torchvision.datasets import CIFAR10, CIFAR100
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from decouple import config
from torch.utils.data import DataLoader, TensorDataset, Dataset
from model import evaluate_model, train_model_camelyon, train_model_cifar, evaluate_model_freeze, evaluate_model_cam_ensemble_freeze, evaluate_model_ensemble_uncertainty
from utils import generate_results, Paths, generate_and_save_plot, bar_plot_diff, block_diff, generate_particles
from preprocessor import load_data_camelyon, load_data_cifar, load_data_imagenet
from src.heads import get_classification_head
from src.linearize import LinearizedImageEncoder
from src.modeling import ImageClassifier, ImageEncoder
from src.linearize import LinearizedImageEncoder
from wilds import get_dataset
from wilds.common.data_loaders import get_train_loader
from bayes_wrap import BayesWrap, generate_freezed_particles, train_model_wrap_cifar, generate_lora_particles
from load_data import ValDataset, TrainDataset
import warnings


warnings.filterwarnings("ignore")


random.seed(2295)

''' -----------------------   Set path ------------------------------'''
paths = Paths(config)
paths.create_path()


''' -----------------------   loading CLIP ViT ------------------------------'''
device = "cuda" if torch.cuda.is_available() else "cpu"

# mdl, preprocess = clip.load('ViT-B/32', device)
mdl, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')


download_path = os.path.expanduser("/media/rokny/DATA1/Afshar/data")
if config('dataset_name').upper() == "CAMELYON":

    dataset = get_dataset(dataset="camelyon17", download=True,  root_dir=download_path)
    train_data = dataset.get_subset(
        "train",
        transform=preprocess
    )

    val_data = dataset.get_subset(
        "val",
        transform=preprocess
    )

    test_data = dataset.get_subset(
        "test",
        transform=preprocess
    )
    print('camelyon loaded')
    trainloaders = [torch.utils.data.DataLoader(train_data, batch_size=int(config('batch_size')), shuffle=True) for i in range(int(config('opt')))]
    valloader = torch.utils.data.DataLoader(val_data, batch_size=int(config('batch_size')), shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=int(config('batch_size')), shuffle=False) 
elif config('dataset_name').upper() == "CIFAR10":

    ''' -----------------------   Loading the Data   ----------------------- '''
    root = os.path.expanduser("/media/rokny/DATA1/Afshar/Data/" + "cifar-10-batches-py")
    train = CIFAR10(root, download=True, train=True)
    test = CIFAR10(root, download=True, train=False, transform=preprocess)
    print(f'len test: {len(test)}')

    print('cifar10 loaded')
    trainloaders, validation_loader, test_loader = load_data_cifar(preprocess, train, test, device)

elif config('dataset_name').upper() == "CIFAR100":

    ''' -----------------------   Loading the Data   ----------------------- '''
    root = os.path.expanduser("/media/rokny/DATA1/Afshar/Data/" + "cifar-100-batches-py")
    train = CIFAR100(root, download=True, train=True)
    test = CIFAR100(root, download=True, train=False, transform=preprocess)

    print('cifar100 loaded')
    trainloaders, validation_loader, test_loader = load_data_cifar(preprocess, train, test, device)

elif config('dataset_name').upper() == "IMAGENET":
    ''' -----------------------   Loading the Data   ----------------------- '''
    train_set = TrainDataset(data_folder='/media/bml/DATA4/Afshar/imagenet/train', transform=preprocess)
    val_dataset = TrainDataset(data_folder='/media/bml/DATA4/Afshar/imagenet/val', transform=preprocess)

    test_dataset = ValDataset(root="/media/bml/DATA4/Afshar/imagenet/test/val", transform=preprocess)
    trainloaders, validation_loader, test_loader = load_data_imagenet( train_set, val_dataset, val_dataset, device)

    print('ImageNet loaded')


# ''' -----------------------   Evaluation the models   ----------------------- '''

mdl, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
# mdl_addr = f'Model/best_model_4_noise_std_0.pt'

particles = generate_freezed_particles(mdl , int(config('opt')))
delta_models = generate_lora_particles(particles)


for i in range(int(config('opt'))):

    mdl_addr = f'Model/best_model_{i}_noise_std_0.pt'
    fine_tuned_weights = torch.load(mdl_addr)
    delta_models[i].load_state_dict(fine_tuned_weights)



noise_std = [0]

all_scores, all_labels= evaluate_model_cam_ensemble_freeze(delta_models, test_loader, device)
all_scores, all_labels= evaluate_model_ensemble_uncertainty(delta_models, test_loader, device)
generate_results(all_scores, all_labels, noise_std[0], i, paths = paths.path_results)