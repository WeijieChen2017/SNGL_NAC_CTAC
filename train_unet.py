from __future__ import print_function

import os
import glob
import json
import numpy as np
from time import time
from matplotlib import pyplot as plt

import tensorflow
from tensorflow.keras.callbacks import History, ModelCheckpoint, TensorBoard, LambdaCallback
from tensorflow.keras.losses import mean_absolute_error, mean_squared_error
from tensorflow.keras.optimizers import Adam

from models import Unet
from utils import NiftiGenerator_opt as NiftiGenerator

para_name = "exper11"
# Data to be written  
train_para ={  
    "para_name" : para_name,
    "img_rows" : 512, # image is resampled to this size
    "img_cols" : 512, # image is resampled to this size
    "channel_X" : 5,
    "channel_Y" : 1,
    "start_ch" : 32,
    "depth" : 3, 
    "validation_split" : 0.2,
    "loss" : "l2",
    "x_data_folder" : 'NPR_SRC', # NAC PET Resampled
    "y_data_folder" : 'CT_SRC',
    "weightfile_name" : 'weights_'+para_name+'.h5',
    "model_name" : 'model_'+para_name+'.json',
    "save_folder" : './save_models/',
    "jpgprogressfile_name" : 'progress_'+para_name,
    "batch_size" : 8, # should be smallish. 1-10
    "num_epochs" : 20, # should train for at least 100-200 in total
    "steps_per_epoch" : 30*20, # should be enough to be equal to one whole pass through the dataset
    "initial_epoch" : 0, # for resuming training
    "load_weights" : False, # load trained weights for resuming training
    "buffer_pool_T" : 30,
    "buffer_pool_V" : 7,
    "learning_rate" : 1e-6,
    "last_layer" : "linear",
    "PET_norm" : "PET_norm_01",
    "CT_norm" : "CT_norm_01",
    "need_norm" : False
}  

for folder_name in ["json", "save_models", "results"]:
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

with open("./json/train_para_"+train_para["para_name"]+".json", "w") as outfile:  
    json.dump(train_para, outfile) 

#######################

def CT_norm_01(data):
    data[data<-1000] = -1000
    data[data>3000] = 3000
    data = (data + 1000) / 4000
    return data

def PET_norm_01(data):
    data[data<0] = 0
    data[data>6000] = 6000
    data = data / 6000
    return data

def CT_norm_m11(data):
    data[data<-1000] = -1000
    data[data>3000] = 3000
    data_mu = np.mean(data)
    data_sigma = np.std(data)
    data = (data - data_mu) / data_sigma
    return data

def PET_norm_m11(data):
    data[data<0] = 0
    data[data>6000] = 6000
    data_mu = np.mean(data)
    data_sigma = np.std(data)
    data = (data - data_mu) / data_sigma
    return data

def train():

    print(train_para)

    np.random.seed(813)
    if train_para["loss"] == "l1":
        loss = mean_absolute_error
    if train_para["loss"] == "l2":
        loss = mean_squared_error

    print('-'*50)
    print('Creating and compiling model...')
    print('-'*50)
    model = Unet.UNetContinuous(img_shape=(train_para["img_rows"],
                                           train_para["img_cols"],
                                           train_para["channel_X"]),
                                out_ch=train_para["channel_Y"],
                                start_ch=train_para["start_ch"],
                                depth=train_para["depth"])
    model.compile(optimizer=Adam(lr=train_para["learning_rate"]),
                  loss=loss,
                  metrics=[mean_squared_error,mean_absolute_error])
    # model.summary()

    # Save the model architecture
    with open(train_para["save_folder"]+train_para["model_name"], 'w') as f:
        f.write(model.to_json())

    # optionally load weights
    if train_para["load_weights"]:
        model.load_weights(train_para["save_folder"]+train_para["weightfile_name"])

    print('-'*50)
    print('Setting up NiftiGenerator')
    print('-'*50)
    niftiGen_augment_opts = NiftiGenerator.PairedNiftiGenerator.get_default_augOptions()
    niftiGen_augment_opts.hflips = True
    niftiGen_augment_opts.vflips = True
    niftiGen_augment_opts.rotations = 15
    niftiGen_augment_opts.scalings = 0
    niftiGen_augment_opts.shears = 0
    niftiGen_augment_opts.translations = 0
    print(niftiGen_augment_opts)
    niftiGen_norm_opts = NiftiGenerator.PairedNiftiGenerator.get_default_normOptions()
    # niftiGen_norm_opts.normXtype = 'fixed'
    # niftiGen_norm_opts.normXoffset = 0
    # niftiGen_norm_opts.normXscale = 6000
    # niftiGen_norm_opts.normYtype = 'fixed'
    # niftiGen_norm_opts.normYoffset = -1000
    # niftiGen_norm_opts.normYscale = 4000
    niftiGen_norm_opts.normXtype = 'function'
    niftiGen_norm_opts.normXfunction = PET_norm_01
    niftiGen_norm_opts.normYtype = 'function'
    niftiGen_norm_opts.normYfunction = CT_norm_01
    niftiGen_norm_opts.needNorm = train_para["need_norm"]
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
                         niftiGen_augment_opts, niftiGen_norm_opts, buffer_pool=train_para["buffer_pool_T"])
    generatorT = niftiGenT.generate(img_size=(train_para["img_rows"],train_para["img_cols"]),
                                    Xslice_samples=train_para["channel_X"],
                                    Yslice_samples=train_para["channel_Y"],
                                    batch_size=train_para["batch_size"])

    niftiGenV = NiftiGenerator.PairedNiftiGenerator()
    niftiGenV.initialize(valid_folderX, valid_folderY,
                         niftiGen_augment_opts, niftiGen_norm_opts, buffer_pool=train_para["buffer_pool_V"] )
    generatorV = niftiGenV.generate(img_size=(train_para["img_rows"],train_para["img_cols"]),
                                    Xslice_samples=train_para["channel_X"],
                                    Yslice_samples=train_para["channel_Y"],
                                    batch_size=train_para["batch_size"])
    print('-'*50)
    print('Preparing callbacks...')
    print('-'*50)
    history = History()
    model_checkpoint = ModelCheckpoint(train_para["save_folder"]+train_para["weightfile_name"],
                                       monitor='val_loss', 
                                       save_best_only=True)
    tensorboard = TensorBoard(log_dir=os.path.join('tblogs','{}'.format(time()), '{}'.format(train_para["para_name"])))
    display_progress = LambdaCallback(on_epoch_end= lambda epoch,
                                      logs: progresscallback_img2img(epoch, logs, model, history, fig, generatorV) )

    print('-'*50)
    print('Fitting network...')
    print('-'*50)
    fig = plt.figure(figsize=(15,5))
    fig.show(False)
    model.fit(generatorT, 
              steps_per_epoch=train_para["steps_per_epoch"],
              epochs=train_para["num_epochs"],
              initial_epoch=train_para["initial_epoch"],
              validation_data=generatorV,
              validation_steps=100,
              callbacks=[history, model_checkpoint, tensorboard] ) # , display_progress

    dataset_go_back(folder_list, sub_folder_list)
    # generatorT.delete_tmp_data()
    # generatorV.delete_tmp_data()
    os.system("mkdir "+train_para["para_name"])
    os.system("mv *"+train_para["para_name"]+"*.jpg "+train_para["para_name"])
    os.system("mv "+train_para["para_name"]+" ./jpeg/"+train_para["para_name"])

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

# Function to display the target and prediction
def progresscallback_img2img(epoch, logs, model, history, fig, generatorV):

    print("-"*50)
    print("Progress evaluation begins:")
    print("-"*50)

    for data in generatorV:
        dataX, dataY = data
        print(dataX.shape, dataY.shape)
        sliceX = dataX.shape[3]
        sliceY = dataY.shape[3]
        break

    print("Prediction generates.")

    predY = model.predict(dataX)
    n_batch = train_para["batch_size"]

    plt.figure(dpi=200)
    for idx in range(n_batch):
        print("subplot ", idx*3+1)
        print("Mean", np.mean(np.squeeze(dataX[idx, :, :, sliceX//2])))
        plt.subplot(n_batch, 3, idx*3+1)
        plt.imshow(np.rot90(np.squeeze(dataX[idx, :, :, sliceX//2])),cmap='gray')
        plt.axis('off')
        plt.title('input X[0]')

        print("subplot ", idx*3+2)
        print("Mean", np.mean(np.squeeze(dataY[idx, :, :, sliceY//2])))
        plt.subplot(n_batch, 3, idx*3+2)
        plt.imshow(np.rot90(np.squeeze(dataY[idx, :, :, sliceY//2])),cmap='gray')
        plt.axis('off')
        plt.title('target Y[0]')

        print("subplot ", idx*3+3)
        print("Mean", np.mean(np.squeeze(predY[idx, :, :, sliceY//2])))
        plt.subplot(n_batch, 3, idx*3+3)
        plt.imshow(np.rot90(np.squeeze(predY[idx, :, :, sliceY//2])),cmap='gray')
        plt.axis('off')
        plt.title('pred. at ' + repr(epoch+1))
    fig.savefig('progress_image_{0}_{1:05d}.jpg'.format(train_para["jpgprogressfile_name"],epoch+1))
    fig.canvas.flush_events()
    print('progress_image_{0}_{1:05d}.jpg saved'.format(train_para["jpgprogressfile_name"],epoch+1))

    plt.figure(dpi=200)
    plt.plot(range(epoch+1),history.history['loss'],'b',label='training loss')
    plt.plot(range(epoch+1),history.history['val_loss'],'r',label='validation loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.yscale('log')
    plt.legend()
    plt.title('Losses')
    fig.tight_layout()
    fig.canvas.draw()
    fig.savefig('progress_loss_{0}_{1:05d}.jpg'.format(train_para["jpgprogressfile_name"],epoch+1))
    fig.canvas.flush_events()
    print('progress_loss_{0}_{1:05d}.jpg saved'.format(train_para["jpgprogressfile_name"],epoch+1))
    plt.close('all')

if __name__ == '__main__':
    train()
