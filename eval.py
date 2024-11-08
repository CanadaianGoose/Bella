# import pickle
# import os
# # import clip
# import open_clip
# import numpy as np
# import copy
# import random
# import ast
# import torch
# import json
# import pandas as pd
# from utils import add_noise_to_parameters
# from src.linearize import LinearizedImageEncoder
# from src.modeling import ImageClassifier, ImageEncoder
# from src.linearize import LinearizedImageEncoder
# from src.heads import get_classification_head
# from bayes_wrap import BayesWrap
# from torchvision.datasets import CIFAR10, CIFAR100
# from decouple import config
# from torch.utils.data import DataLoader, TensorDataset
# from torchvision.datasets import CIFAR10
# from model import evaluate_model_freeze, evaluate_cam_mix_ensemble_soup, evaluate_model_cam_ensemble_freeze, averaging_model, best_combination
# from utils import generate_results, Paths, generate_mix_plot_acc, generate_mix_plot_ent, calc_confusion_uncertainty_matrix
# from preprocessor import load_data_camelyon, load_data_cifar, load_data_places#, CorruptedCIFARDataset
# from wilds import get_dataset
# from wilds.common.data_loaders import get_train_loader
from collections import defaultdict
from torch.nn import CosineSimilarity

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
# import deeplake
import torch.nn.functional as F
from torchvision.datasets import CIFAR10, CIFAR100
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from decouple import config
from torch.utils.data import DataLoader, TensorDataset, Dataset
from torchvision.datasets import CIFAR10
from model import evaluate_model, train_model_camelyon, train_model_cifar, evaluate_model_freeze, evaluate_model_cam_ensemble_freeze, evaluate_cam_mix_ensemble_soup
from utils import generate_results, Paths, generate_and_save_plot, bar_plot_diff, block_diff, generate_particles
from preprocessor import load_data_camelyon, load_data_cifar, load_data_imagenet#, load_data_places
from src.heads import get_classification_head
from src.linearize import LinearizedImageEncoder
from src.modeling import ImageClassifier, ImageEncoder
from src.linearize import LinearizedImageEncoder
from wilds import get_dataset
from load_data import ValDataset, TrainDataset
from wilds.common.data_loaders import get_train_loader
from bayes_wrap import BayesWrap, generate_freezed_particles, train_model_wrap_cifar, generate_lora_particles

# from diversity_measurement import diversity_measure


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

    # corrupted_testset = np.load("Data/speckle_noise.npy")
    # lbls = np.load("Data/labels.npy")
    # test.data = corrupted_testset
    # test.targets = lbls
    # test.transform = preprocess

    print(f'len test zahra: {len(test)}')
    print('cifar10 loaded')
    trainloaders, validation_loader, test_loader = load_data_cifar(preprocess, train, test, device)

elif config('dataset_name').upper() == "CIFAR100":

    ''' -----------------------   Loading the Data   ----------------------- '''
    root = os.path.expanduser("/media/rokny/DATA1/Afshar/Data/" + "cifar-100-batches-py")
    train = CIFAR100(root, download=True, train=True)
    test = CIFAR100(root, download=True, train=False, transform=preprocess)


    print('cifar100 loaded')
    trainloaders, validation_loader, test_loader = load_data_cifar(preprocess, train, test, device)

elif config('dataset_name').upper() == "PLACES205":

    ''' -----------------------   Loading the Data   ----------------------- '''
    ds = deeplake.load("hub://activeloop/places205")
    total_samples = len(ds)
    train_split = int(0.7 * total_samples)
    val_split = int(0.1 * total_samples)
    train_data = ds[:train_split]
    val_data = ds[train_split:train_split + val_split]
    test_data = ds[train_split + val_split:]
    print('Places205 has been loaded')
    print(f'len train is {len(train_data)}')
    print(f'len train is {len(val_data)}')
    print(f'len train is {len(test_data)}')
    trainloaders, validation_loader, test_loader = load_data_places(preprocess, train_data, val_data, test_data, device)


elif config('dataset_name').upper() == "DOMAINNET":

    ''' -----------------------   Loading the Data   ----------------------- '''
    train_data = deeplake.load("hub://activeloop/domainnet-real-train")
    test_data = deeplake.load("hub://activeloop/domainnet-real-test")

    print('Domainnet has been loaded')
    print(f'len train is {len(train_data)}')
    print(f'len test is {len(test_data)}')

    trainloaders, validation_loader, test_loader = load_data_places(preprocess, train_data, test_data, test_data, device)

elif config('dataset_name').upper() == "IMAGENET":
    ''' -----------------------   Loading the Data   ----------------------- '''
    train_set = TrainDataset(data_folder='/media/bml/DATA4/Afshar/imagenet/train', transform=preprocess)
    val_dataset = TrainDataset(data_folder='/media/bml/DATA4/Afshar/imagenet/val', transform=preprocess)

    test_dataset = ValDataset(root="/media/bml/DATA4/Afshar/imagenet/test/val", transform=preprocess)
    trainloaders, validation_loader, test_loader = load_data_imagenet( train_set, val_dataset, val_dataset, device)

    print('ImageNet loaded')

# ''' -----------------------   Evaluating the model   ----------------------- '''
mdl, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
model_address = [f for f in os.listdir('/media/bml/DATA4/Afshar/set/imgnt_lora/') if f[-3:]=='.pt']

# # # # /media/rokny/DATA4/Afshar/new_cifar102/
print(f'number of checkpoints is {len(model_address)}')



# # sssddd
# # a = 0.6
# # average_model, ensemble = averaging_model(model_address)
# # noise_std = [0]
# # i=0
# # all_scores, all_labels = evaluate_model_cam_ensemble_freeze(ensemble, test_loader, device)
# # corr_acc = generate_results(all_scores, all_labels, noise_std[0], i, paths = paths.path_results)
# # ssdddd
# # noise_std = [0]
# # i=0

# # nn = [i / 10.0 for i in range(1, 7)]
# # mm = [i / 10.0 for i in range(1, 7)]
# # kk = [i / 10.0 for i in range(1, 7)]

# # init_accuracy = 0.94
# # for n in nn:
# #     for m in mm:
# #         for k in kk:

# #             average_model, _ = averaging_model(model_address, n, m, k)
# #             all_scores, all_labels = evaluate_model_freeze(average_model, val_loader, device)
# #             accuracy = generate_results(all_scores, all_labels, noise_std[0], i, paths = paths.path_results)
# #             if accuracy >= init_accuracy:
# #                 init_accuracy = accuracy
# #                 best_n = n
# #                 best_m = m
# #                 best_k = k

# # print(f"best values for n,m,k are {best_n}, {best_m}, {best_k}")

# # average_model, ensemble = averaging_model(model_address, 0.6, 0.1, 0.3)
# # all_scores, all_labels = evaluate_model_freeze(average_model, val_loader, device)
# # accuracy = generate_results(all_scores, all_labels, noise_std[0], i, paths = paths.path_results)

# # sssdddd

# def greedy_soup(ens_addr, validation_loader):
#     ensemble=[]
#     for i, addrr in enumerate(ens_addr):
#         mdl, _, _ = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
#         # mdl_addr = f'mdl-cam3/best_model_{i}_noise_std_0_series_0.pt'

#         classification_head = get_classification_head()
#         image_encoder = ImageEncoder(mdl)#, keep_lang=False)
#         NET = ImageClassifier(image_encoder, classification_head)
#         NET.freeze_head()

#         model_new = copy.deepcopy(NET)
#         fine_tuned_weights = torch.load('/media/rokny/DATA4/Afshar/new_cifar102/' + addrr)
#         # fine_tuned_weights = torch.load("mdl-svgd-gold/" + addrr)

#         model_new.load_state_dict(fine_tuned_weights)

#         ensemble.append(model_new)
#         print(f'model {i} is loaded from {addrr}')

#     print(f'number of models for soup is {len(ensemble)}')
#     model, _, _ = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
#     classification_head = get_classification_head()
#     image_encoder = ImageEncoder(model)#, keep_lang=False)
#     NET = ImageClassifier(image_encoder, classification_head)

#     average_model = copy.deepcopy(NET)
#     state_dicts = [mdel.state_dict() for mdel in ensemble]

#     average_state_dict = {}
#     num_models = len(ensemble)



#     for key in ensemble[0].state_dict():
#         average_state_dict[key] =sum(state_dict[key] for state_dict in state_dicts) / num_models

#     average_model.load_state_dict(average_state_dict)

#     print('The averaged model will be used for comparison')

#     noise_std = [0]
   
#     all_scores, all_labels, all_entrpoies = evaluate_model_freeze(average_model, validation_loader, device)
#     UA = calc_confusion_uncertainty_matrix(all_scores, all_labels, all_entrpoies, pe_entropy_thresh = 0.4)
#     accuracy = generate_results(all_scores, all_labels, noise_std[0], i, paths = paths.path_results)
#     # print(f'accuracy is {accuracy}')
#     # print(f'ua is {UA}')

#     return accuracy, UA

# ens_addr=[]
# mdl1_address = "mdl_1.pt"
# ens_addr.append(mdl1_address)
# ua_addr = []
# total_accs=[]
# for j in range(4):

#     init_acc = 0.9
#     init_ua = 0.9
#     accuracies , ent=[], []
#     for i,adr in enumerate(model_address):
        
#         if adr not in ens_addr:
#             ens_addr.append(adr)

#             accuracy, ua = greedy_soup(ens_addr, test_loader)
#             # ent.append(dif_ent)
#             accuracies.append(accuracy)
#             # if ua > init_ua:
#             #     init_ua = ua
#             #     best_ua_adr = adr
#             #     print(f'best mse for {adr} with value {init_ua} has been saved')
    
#             if accuracy> init_acc:
#                 init_acc = accuracy
#                 best_comb = adr
#                 print(f'best model has been found with accuracy {init_acc}')
#             ens_addr.pop()
#     print(f"best model address is {best_comb} for set of {j+2} with accuracy {init_acc}")
#     print(f" all accuracies for set of {j+2} are {accuracies}")
#     # print(f'best mse is {init_ent} for {best_ent_adr}')
#     # print(f'all mse for set of {j+2} are {ent}')
#     total_accs.append(accuracies)
#     ens_addr.append(best_comb)
#     # ua_addr.append(best_ua_adr)
#     print(f"len ens_adr is {len(ens_addr)}")
#     # print(f"len ua_adr is {len(ua_addr)}")


# acc_info = {  "set of two": total_accs[0],
#                     "set of three": total_accs[1],   
#                     "set of four":  total_accs[2],
#                     "set of five": total_accs[3]
#                                             }

# acc_info_path = f"Results/acc_info_cifar100.json"
# with open(acc_info_path, 'w') as fp:
#     json.dump(acc_info, fp, indent=2)
# print(f'ua addresses is {ua_addr}')
# sssddd


# my_dict = block_diff(model, average_model)
# bar_plot_diff(my_dict, noise_std_list, paths.path_figure)
# df = df.append(my_dict, ignore_index=True)



# df.to_csv(paths.path_results +'results_noise_{noise_std_list}.csv' , index = False)


###################################################################################################

# # ''' -----------------------   Evaluating the model individualy   ----------------------- '''

# ''' -----------------------   Evaluating the model   ----------------------- '''

## df = pd.DataFrame(columns=['visual', 'token_embedding', 'ln_final'])
# mdl, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')

# model_address = [f for f in os.listdir('./nmdl/') if f[-3:]=='.pt']

# print(f'number of checkpoints is {len(model_address)}')
# ensemble=[]
# for i in range(5):
#     mdl, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
#     # mdl_addr = f'nmdl/mdl_{i}.pt'

#     classification_head = get_classification_head()
#     image_encoder = ImageEncoder(mdl)#, keep_lang=False)
#     net = ImageClassifier(image_encoder, classification_head)
#     net.freeze_head()

#     model_new = copy.deepcopy(net)
#     fine_tuned_weights = torch.load("./nmdl/"+ model_address[i])
#     model_new.load_state_dict(fine_tuned_weights)

#     ensemble.append(model_new)
#     print(f'model {i} is loaded from {model_address[i]}')


# model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
# classification_head = get_classification_head()
# image_encoder = ImageEncoder(model)#, keep_lang=False)
# net = ImageClassifier(image_encoder, classification_head)

# average_model = copy.deepcopy(net)
# state_dicts = [mdel.state_dict() for mdel in ensemble]

# average_state_dict = {}
# num_models = len(ensemble)

# # coefficients = [0.9250, 0.8823, 0.9373, 0.9393, 0.9273]


# for key in ensemble[0].state_dict():
#     average_state_dict[key] =sum(state_dict[key] for state_dict in state_dicts) / num_models
#     # average_state_dict[key] = sum([coeff * state_dict[key] for coeff, state_dict in zip(coefficients, state_dicts)])/ len(coefficients)

# average_model.load_state_dict(average_state_dict)

# print('The averaged model will be used for comparison')
# print("")
# # noise_std = [0]
# # all_scores, all_labels = evaluate_model_freeze(average_model, test_loader, device)
# # generate_results(all_scores, all_labels, noise_std[0], i, paths = paths.path_results)

# # ''' -----------------------   Evaluating the model for entropies   ----------------------- '''

# # df = pd.DataFrame(columns=['visual', 'token_embedding', 'ln_final'])
# # noise_std_list = [0 for i in range(5)]
# # noise_std_list = ast.literal_eval(config('noise_std_list'))
# # models_address = ["./Model-5nosplit-cam-simple" + f"/best_model_{i}.pt" for i,j in enumerate(noise_std_list)]

# ensemble = []
# model_address = [f for f in os.listdir('./nmdl/') if f[-3:]=='.pt']
# for i in range(5):
#     mdl, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
    
#     classification_head = get_classification_head()
#     image_encoder = ImageEncoder(mdl)#, keep_lang=False)
#     NET = ImageClassifier(image_encoder, classification_head)
#     NET.freeze_head()

#     model_new = copy.deepcopy(NET)
#     fine_tuned_weights = torch.load("./nmdl/" + model_address[i])
#     model_new.load_state_dict(fine_tuned_weights)

#     ensemble.append(model_new)
#     print(f'model {i} is loaded from {model_address[i]}')


# print(f'len ensemble is {len(ensemble)}')
# print('hi Zahra')
# optimizing_a(ensemble, average_model, test_loader, text_inputs, device)
# noise_std=[0]

# all_scores, all_labels = evaluate_model_cam_ensemble_freeze(ensemble, test_loader, device)
# generate_results(all_scores, all_labels, noise_std[0], i, paths = paths.path_results)







#############################################################################################

# noise_std_list = [0]
# total_acc, total_ent = [], []
# model_address = [f for f in os.listdir('nmdl/') if f[-3:]=='.pt']
# print(model_address)
# # for j in range(10):
#     # selected_models = random.sample(model_address, 5)
#     # print(selected_models)
# average_model, ensemble = averaging_model(model_address)
# a = [i * 0.1 for i in range(11)]
# # print(a)
# # a=[0.7]
# all_acc, all_ent = [],[]
# for i,x in enumerate(a):
#     all_scores, all_labels, ent = evaluate_cam_mix_ensemble_soup(ensemble, average_model, x, test_loader, device)
#     accuracy =  generate_results(all_scores, all_labels, noise_std_list[0], i, paths = paths.path_results)
#     all_acc.append(accuracy)
#     all_ent.append(ent)
#     total_acc.append(all_acc)


# print(f'all_acc is {all_acc}')
# all_acc = f"Results/acc_cifar10_nset.json"
# with open(total_acc, 'w') as fp:
#     json.dump(total_acc, fp, indent=2)



# print(f"accuracies are: {all_acc}")
# print(f"entropies are: {all_ent}")

# save_path_acc = paths.path_results + "mix_acc.png"
# save_path_ent = paths.path_results + "mix_ent.png"
# generate_mix_plot_acc(all_acc, a, save_path_acc)
# generate_mix_plot_ent(all_ent, a, save_path_ent)

# sssddd
#############################################################################################

# mdl, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')


# i = 0
# mdl_addr = '/media/rokny/DATA4/Afshar/cifar10_final/mdl_1.pt'

# classification_head = get_classification_head()
# image_encoder = ImageEncoder(mdl)#, keep_lang=False)
# NET = ImageClassifier(image_encoder, classification_head)
# NET.freeze_head()

# model_new = copy.deepcopy(NET)
# fine_tuned_weights = torch.load(mdl_addr)
# model_new.load_state_dict(fine_tuned_weights)

# print(f'model {i} is loaded from {mdl_addr}')


# # opt = config('opt')
# # model = BayesWrap(NET, opt)



# model_new = model_new.cuda()

# # average_model, ensemble = averaging_model(model_address)

# noise_std = [0]


# all_scores, all_labels, all_entrpoies = evaluate_model_freeze(model_new, validation_loader, device)
# all_ua=[]
# for l in [0.1, 0.2, 0.25, 0.3, 0.4, 0.5]:
#     UA = calc_confusion_uncertainty_matrix(all_scores, all_labels, all_entrpoies, pe_entropy_thresh = l)
#     all_ua.append(UA)
# accuracy = generate_results(all_scores, all_labels, noise_std[0], i, paths = paths.path_results)
# print(f'accuracy is {accuracy}')
# print(f'all_ua is {all_ua}')

#####################################################################################################


# mdl, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
# model_address = [f for f in os.listdir('./nmdl/') if f[-3:]=='.pt']
# print(f'number of checkpoints is {len(model_address)}')

# average_model, ensemble = averaging_model(model_address)
# noise_std = [0]
# i = 0
# a= 0.6

# corrupted_address = [f for f in os.listdir("./cifar100-c/") if f[-4:]=='.npy']

# # print(f'corrupted is {corrupted_address}')
# performance=[]
# for i, corr in enumerate(corrupted_address):
#     if corr != "labels.npy":
#         print(f'noise is {corr.split(".")[0]}')
#         perf=[]
#         corrupted_testset = np.load("cifar100-c/" + corr)
#         lbls = np.load("cifar100-c/labels.npy")
#         test.data = corrupted_testset
#         test.targets = lbls
#         test.transform = preprocess
#         print(f'len {corr} is {len(test)}')
#         trainloaders, validation_loader, test_loader = load_data_cifar(preprocess, train, test, device)
#         perf.append(corr.split(".")[0])

#         all_scores, all_labels,_ = evaluate_model_freeze(average_model, test_loader, device)
#         accuracy = generate_results(all_scores, all_labels, noise_std[0], i, paths = paths.path_results)
#         print(" ")
#         perf.append(accuracy)

#         all_scores, all_labels,_ = evaluate_cam_mix_ensemble_soup(ensemble, average_model, a, test_loader, device)
#         mix_acc = generate_results(all_scores, all_labels, noise_std[0], i, paths = paths.path_results)
#         perf.append(mix_acc)


#         all_scores, all_labels = evaluate_model_cam_ensemble_freeze(ensemble, test_loader, device)
#         corr_acc = generate_results(all_scores, all_labels, noise_std[0], i, paths = paths.path_results)
#         perf.append(corr_acc)
#         print(f"noise: {corr.split('.')[0]} has been finished")
#         print(" ")
  
#         performance.append(perf)
    
    
    
# performance_path = f"Model/salp_oens_cifar100-nc.json"
# with open(performance_path, 'w') as fp:
#     json.dump(performance, fp, indent=2)
# print(performance)


#-------------------------------------------------------------------------------------------------

# noise_std = [0]
# i=0

# nn = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]
# mm = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]
# kk = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]
# pp = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]
# tt = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]

# perf = []
# init_accuracy = 0.85
# for n in nn:
#     for m in mm:
#         for k in kk:
#             for p in pp:
#                 for t in tt:
#                     if n+m+k+p+t==1:
#                         all_scores, all_labels = evaluate_model_cam_ensemble_freeze(ensemble, test_loader, n, m, k, p, t, device)
#                         accuracy = generate_results(all_scores, all_labels, noise_std[0], i, paths = paths.path_results)
#                         if accuracy >= init_accuracy:
#                             init_accuracy = accuracy
#                             best_n = n
#                             best_m = m
#                             best_k = k
#                             best_p = p
#                             best_t = t
#                             per = [best_n, best_m, best_k, best_p, best_t, init_accuracy]
#                             print(f'the combination has found: {per}')
#                             perf.append(per)

# print(f"best values for n,m,k,p,t are {best_n}, {best_m}, {best_k}, {best_p}, {best_t} with accuracy of {init_accuracy}")
# print(perf)

#-------------------------------------------------------------------------------------------------------------------


# model_address = [f for f in os.listdir('nmdl/') if f[-3:]=='.pt']
# print(model_address)

# # average_model, ensemble = averaging_model(model_address)
# # print(len(ensemble))

# data_path='Results/data.pt'

# if os.path.exists(data_path):
#     # Load the tensor from the file if it exists
#     data = torch.load(data_path)

# else:    
#     average_model, ensemble = averaging_model(model_address)
#     ensemble_outputs = []
#     for model in ensemble:
#         model.to(device)
#         model.eval()
#         model_outputs = []  

#         with torch.no_grad():
#             for inputs, _ in test_loader:
#                 inputs = inputs.to(device)
#                 l = model(inputs)  
#                 outputs= torch.softmax(l, 1)
#                 for output in outputs:
#                     model_outputs.append(output.cpu().tolist()) 

#         ensemble_outputs.append(model_outputs)

#     data = torch.tensor([list(sample) for sample in zip(*ensemble_outputs)])
#     torch.save(data, data_path)



# # print(f'data.shape: {data.shape}')
# # print(data[0])

# diversity={}
# diversity_path='Results/diversity.json'
# modes=['kl_div', 'js_div', 'class_var']
# for mode in modes:
#     final = diversity_measure(mode,data)
#     diversity[mode] = final.cpu().tolist()


# with open(diversity_path, 'w') as json_file:
#     json.dump(diversity, json_file, indent=2)

#-------------------------------------------------------------------------------------------------------------------
particles = generate_freezed_particles(mdl , int(config('opt')))
delta_models = generate_lora_particles(particles)

def load_ensemble_lora(ens_addr, delta_models):

    for i, addrr in enumerate(ens_addr):

        # mdl_addr = f'/media/rokny/DATA4/Afshar/Lora_paper/mdl-lora-50p/{addrr}'
        mdl_addr = f'Model/{addrr}'
        fine_tuned_weights = torch.load(mdl_addr)
        delta_models[i].load_state_dict(fine_tuned_weights)


        print(f'model {i} is loaded from {addrr}')

    # print(f'number of models loaded is {len(ensemble)}')
    return delta_models



def compute_similarity(models, device):

    similarity = {}
    similarity_mean = []
    l2_dist = 0.0
    cos_fun = CosineSimilarity(dim=0, eps=1e-6)
    for idx, params in enumerate(zip(models[0].parameters(), models[1].parameters())):
        p1, p2 = params
        p1 = p1.to(device)
        p2 = p2.to(device)

        clean_param1 = torch.nan_to_num(p1, nan=0.0).flatten()
        clean_param2 = torch.nan_to_num(p2, nan=0.0).flatten()
  

        l2_dist += torch.linalg.norm(clean_param1 - clean_param2, ord=2)

        similarity[idx] = torch.clamp(torch.nan_to_num(cos_fun(p1.flatten(), p2.flatten()), nan=0), min=0.0, max=1.0)
        # print(f'sim is {similarity[idx]}')
        # print(f' L2: {l2_dist.item()}')
        similarity_mean.append(torch.mean(similarity[idx]))

    # print(torch.mean(torch.stack(similarity_mean)))
    # print(f' L2: {l2_dist.item()}')
    return l2_dist.item()




# ens_addr=[]
# mdl1_address = "mdl_1.pt"
# ens_addr.append(mdl1_address)
# ua_addr = []
# total_accs=[]

# init_dist=0.1

# initial_dist_min = 500

# accuracies , ent=[], []
# for i,adr in enumerate(model_address):
    
#     if adr not in ens_addr:
#         ens_addr.append(adr)
#         models = load_ensemble_lora(ens_addr, delta_models)

#         dist = compute_similarity(models, device)


#         if dist > init_dist:
#             init_dist = dist
#             max_dist_adr = adr
#             print(f'max dist: {max_dist_adr}, value: {dist}')

#         if dist < initial_dist_min:
#             initial_dist_min = dist
#             min_dist_adr = adr
#             print(f'min dist: {min_dist_adr}, value: {dist}')
#         ens_addr.pop()

# print(f'max_dist_adr is {max_dist_adr}')
# print(f'min_dist_adr is {min_dist_adr}')


noise_std_list = [0]
total_acc, total_ent = [], []
model_address = [f for f in os.listdir('Model/') if f[-4:]=='1.pt']
print(model_address)

ensemble = load_ensemble_lora(model_address, delta_models)

a = [i * 0.1 for i in range(11)]
# print(a)
# a=[0.7]
all_acc, all_ent = [],[]
for i,x in enumerate(a):
    all_scores, all_labels, ent = evaluate_cam_mix_ensemble_soup([ensemble[0]], ensemble[1], x, test_loader, device)
    accuracy =  generate_results(all_scores, all_labels, noise_std_list[0], i, paths = paths.path_results)
    all_acc.append(accuracy)
    all_ent.append(ent)
    # total_acc.append(all_acc)


print(f'all_acc is {all_acc}')
print(f'all_ent is {all_ent}')

# all_acc = f"Results/acc_cifar10_nset.json"
# with open(total_acc, 'w') as fp:
#     json.dump(total_acc, fp, indent=2)

