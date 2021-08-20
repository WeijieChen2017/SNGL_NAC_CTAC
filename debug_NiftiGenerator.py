from __future__ import print_function

import os
import glob
import json
import numpy as np
from time import time
from utils import NiftiGenerator

def CT_norm(data):
    data[data<-1000] = -1000
    data[data>3000] = 3000
    data = (data + 1000) / 4000
    return data

def dataset_go_back(folder_list, sub_folder_list):

    [folderX, folderY] = folder_list
    [train_folderX, train_folderY, valid_folderX, valid_folderY] = sub_folder_list
    
    data_trainX_list = glob.glob(train_folderX+"/*.nii")+glob.glob(train_folderX+"/*.nii.gz")
    data_validX_list = glob.glob(valid_folderX+"/*.nii")+glob.glob(valid_folderX+"/*.nii.gz")
    data_trainY_list = glob.glob(train_folderY+"/*.nii")+glob.glob(train_folderY+"/*.nii.gz")
    data_validY_list = glob.glob(valid_folderY+"/*.nii")+glob.glob(valid_folderY+"/*.nii.gz")

    for data_path in data_trainX_list:
        cmd = "mv "+data_path+" "+folderX
        os.system(cmd)

    for data_path in data_validX_list:
        cmd = "mv "+data_path+" "+folderX
        os.system(cmd)

    for data_path in data_trainY_list:
        cmd = "mv "+data_path+" "+folderY
        os.system(cmd)

    for data_path in data_validY_list:
        cmd = "mv "+data_path+" "+folderY
        os.system(cmd)

# Split the dataset and move them to the corresponding folder
def split_dataset(folderX, folderY, validation_ratio):

    train_folderX = folderX + "/trainX/"
    train_folderY = folderY + "/trainY/"
    valid_folderX = folderX + "/validX/"
    valid_folderY = folderY + "/validY/"

    for folder_name in [train_folderX, train_folderY, valid_folderX, valid_folderY]:
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)

    data_path_list = glob.glob(folderX+"/*.nii") + glob.glob(folderX+"/*.nii.gz")
    data_path_list.sort()
    print(data_path_list)
    data_path_list = np.asarray(data_path_list)
    np.random.shuffle(data_path_list)
    data_path_list = list(data_path_list)
    data_name_list = []
    for data_path in data_path_list:
        data_name_list.append(os.path.basename(data_path))

    valid_list = data_name_list[:int(len(data_name_list)*validation_ratio)]
    valid_list.sort()
    train_list = list(set(data_name_list) - set(valid_list))
    train_list.sort()

    print("valid_list: ", valid_list)
    print('-'*50)
    print("train_list: ", train_list)

    for valid_name in valid_list:
        valid_nameX = folderX+"/"+valid_name
        valid_nameY = folderY+"/"+valid_name.replace("NPR", "CT")
        cmdX = "mv "+valid_nameX+" "+valid_folderX
        cmdY = "mv "+valid_nameY+" "+valid_folderY
        print(cmdX)
        print(cmdY)
        os.system(cmdX)
        os.system(cmdY)

    for train_name in train_list:
        train_nameX = folderX+"/"+train_name
        train_nameY = folderY+"/"+train_name.replace("NPR", "CT")
        cmdX = "mv "+train_nameX+" "+train_folderX
        cmdY = "mv "+train_nameY+" "+train_folderY
        print(cmdX)
        print(cmdY)
        os.system(cmdX)
        os.system(cmdY)

    return [train_folderX, train_folderY, valid_folderX, valid_folderY]


para_name = "exper01"
# Data to be written  
train_para ={  
    "para_name" : para_name,
    "img_rows" : 512, # image is resampled to this size
    "img_cols" : 512, # image is resampled to this size
    "channel_X" : 5,
    "channel_Y" : 1,
    "start_ch" : 64,
    "depth" : 4, 
    "validation_split" : 0.2,
    "loss" : "l1",
    "x_data_folder" : 'NPR_SRC', # NAC PET Resampled
    "y_data_folder" : 'CT_SRC',
    "weightfile_name" : 'weights_'+para_name+'.h5',
    "model_name" : 'model_'+para_name+'.json',
    "save_folder" : './save_models/',
    "jpgprogressfile_name" : 'progress_'+para_name,
    "batch_size" : 2, # should be smallish. 1-10
    "num_epochs" : 3, # should train for at least 100-200 in total
    "steps_per_epoch" : 30, # should be enough to be equal to one whole pass through the dataset
    "initial_epoch" : 0, # for resuming training
    "load_weights" : False, # load trained weights for resuming training
}  

for folder_name in ["json", "save_models", "results"]:
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

print(train_para)

np.random.seed(813)

print('-'*50)
print('Setting up NiftiGenerator')
print('-'*50)
niftiGen_augment_opts = NiftiGenerator.PairedNiftiGenerator.get_default_augOptions()
niftiGen_augment_opts.hflips = True
niftiGen_augment_opts.vflips = True
niftiGen_augment_opts.rotations = 15
niftiGen_augment_opts.scalings = 0.25
niftiGen_augment_opts.shears = 0
niftiGen_augment_opts.translations = 10
print(niftiGen_augment_opts)
niftiGen_norm_opts = NiftiGenerator.PairedNiftiGenerator.get_default_normOptions()
niftiGen_norm_opts.normXtype = 'fixed'
niftiGen_norm_opts.normXoffset = 0
niftiGen_norm_opts.normXscale = 6000
# niftiGen_norm_opts.normYtype = 'fixed'
# niftiGen_norm_opts.normYoffset = -1000
# niftiGen_norm_opts.normYscale = 4000
niftiGen_norm_opts.normYtype = 'function'
niftiGen_norm_opts.normYfunction = CT_norm
print(niftiGen_norm_opts)

folderX = "./data_train/"+train_para["x_data_folder"]
folderY = "./data_train/"+train_para["y_data_folder"]
folder_list = [folderX, folderY]
sub_folder_list = split_dataset(folderX=folderX, folderY=folderY, 
                                validation_ratio=train_para["validation_split"])
[train_folderX, train_folderY, valid_folderX, valid_folderY] = sub_folder_list
print(train_folderX, train_folderY, valid_folderX, valid_folderY)

niftiGenT = NiftiGenerator.PairedNiftiGenerator()
niftiGenT.initialize(train_folderX, train_folderY,
                     niftiGen_augment_opts, niftiGen_norm_opts)
# generatorT = niftiGenT.generate(img_size=(train_para["img_rows"],train_para["img_cols"]),
#                                 Xslice_samples=train_para["channel_X"],
#                                 Yslice_samples=train_para["channel_Y"],
#                                 batch_size=train_para["batch_size"])

niftiGenV = NiftiGenerator.PairedNiftiGenerator()
niftiGenV.initialize(valid_folderX, valid_folderY,
                     niftiGen_augment_opts, niftiGen_norm_opts )
# generatorV = niftiGenV.generate(img_size=(train_para["img_rows"],train_para["img_cols"]),
#                                 Xslice_samples=train_para["channel_X"],
#                                 Yslice_samples=train_para["channel_Y"],
#                                 batch_size=train_para["batch_size"])

dataset_go_back(folder_list, sub_folder_list)
# generatorT.delete_tmp_data()
# generatorV.delete_tmp_data()