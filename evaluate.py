from __future__ import print_function

import os
#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import glob
import json
from time import time
import nibabel
import numpy as np
from keras.models import model_from_json
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, TensorBoard
# from keras.optimizers import Adam
from keras.losses import mean_absolute_error, mean_squared_error
from keras import backend as K

from utils import dataUtilities as du
from utils import NiftiGenerator
from train_unet import PET_norm_m11, CT_norm_m11

def eval():
    train_para_name_hub = ["exper06"]
    test_para_name_prefix = "exper"
    test_count = 6
    test_count -= 1 # for iteration begining, it add by 1 in the first iteration.

    for train_para_name in train_para_name_hub:
        test_count += 1
        test_para_name = test_para_name_prefix + "{0:0>2}".format(test_count)

        with open("./json/train_para_"+train_para_name+".json") as f:
          train_para = json.load(f)

        test_para ={  
            "test_para_name" : test_para_name,
            "train_para_name" : train_para_name,
            "channel_X" : train_para["channel_X"],
            "channel_Y" : train_para["channel_Y"], 
            "data_folder" : 'NAC',
            "img_rows": 512,
            "img_cols": 512,
            "batch_size" : 4
        }

        print("Model: ./save_models/model_"+test_para["train_para_name"]+".json")
        model_list = glob.glob("./save_models/model_"+test_para["train_para_name"]+".json")
        model_list.sort()
        for model_path in model_list:
            weights_path = "./save_models/weights_"+test_para["train_para_name"]+".h5"
            print('-'*50)
            print('Loading model...')
            print('-'*50)
            print("Model path: ", model_path)
            print("Weights path: ", weights_path)
        
            with open(model_path, 'r') as f:
                model = model_from_json(f.read())
            model.load_weights(weights_path)

            print('-'*50)
            print('Loading images from {}...'.format(test_para["data_folder"]))
            print('-'*50)


            # niftiGen_augment_opts = NiftiGenerator.SingleNiftiGenerator.get_default_augOptions()
            # niftiGen_augment_opts.hflips = False
            # niftiGen_augment_opts.vflips = False
            # niftiGen_augment_opts.rotations = 0
            # niftiGen_augment_opts.scalings = 0
            # niftiGen_augment_opts.shears = 0
            # niftiGen_augment_opts.translations = 0
            # print(niftiGen_augment_opts)
            # niftiGen_norm_opts = NiftiGenerator.SingleNiftiGenerator.get_default_normOptions()
            # niftiGen_norm_opts.normXtype = 'none'
            # niftiGen_norm_opts.normYtype = 'none'
            # print(niftiGen_norm_opts)


            testX_list = glob.glob("./data_test/"+test_para["data_folder"]+"/NPR_*.nii")
            testX_list += glob.glob("./data_test/"+test_para["data_folder"]+"/NPR_*.nii.gz")
            print("./data_test/"+test_para["data_folder"]+"/*_NPR.nii.gz")
            testX_list.sort()
            for testX_path in testX_list:
                print("testX: ", testX_path)
                gt_path = testX_path.replace("NPR", "CT")
                gt_data = nibabel.load(gt_path).get_fdata()
                gt_data = CT_norm_m11(gt_data)

                testX_name = os.path.basename(testX_path)
                testX_file = nibabel.load(testX_path)
                testX_data = testX_file.get_fdata()
                testX_data = PET_norm_m11(testX_data)
                # inputX = np.transpose(testX_norm, (2,0,1))

                # niftiGenE = NiftiGenerator.SingleNiftiGenerator()
                # test_folderX = "./data_test/"+test_para["data_folder"]
                # niftiGenE.initialize(test_folderX, niftiGen_augment_opts, niftiGen_norm_opts)
                # generatorE = niftiGenE.generate(img_size=(test_para["img_rows"],test_para["img_cols"]),
                                                # slice_samples=test_para["channel_X"],
                                                # batch_size=test_para["batch_size"])
                inputX = createInput(testX_data, n_slice=test_para["channel_X"])
                # print("inputX shape: ", inputX.shape)
                outputY = np.zeros(testX_data.shape)
                for idx in range(testX_data.shape[2]):
                    print("-"*50)
                    inputX_slice = inputX[idx, :, :, :].reshape(1, 
                                                                test_para["img_rows"],
                                                                test_para["img_cols"],
                                                                test_para["channel_X"])
                    # print("inputX_slice shape: ", inputX_slice.shape)
                    # print("inputX_slice mean:", np.mean(inputX_slice))
                    # print("inputX_slice std:", np.std(inputX_slice))
                    outputY_slice =  model.predict(inputX_slice, verbose=1)
                    # print("outputY_slice shape: ", outputY_slice.shape)
                    # print("outputY_slice mean:", np.mean(outputY_slice))
                    # print("outputY_slice std:", np.std(outputY_slice))
                    outputY[:, :, idx] = np.squeeze(np.transpose(outputY_slice, (1,2,0,3))[:, :, :, test_para["channel_Y"] // 2])
                    # if idx == 32:
                    #     np.save(testX_name+"_inputX.npy", inputX_slice)
                    #     np.save(testX_name+"_outputY.npy", outputY_slice)


                # np.save(testX_name+"_inputX.npy", inputX)
                # print("inputX shape: ", inputX.shape)
                # outputY =  model.predict(generatorE, verbose=1)

                # print("outputY shapels: ", outputY.shape)
                predY_data = outputY
                # predY_data[predY_data < 0] = 0
                # testX_sum = np.sum(testX_data)
                # predY_sum = np.sum(predY_data)
                # predY_data = predY_data / predY_sum * testX_sum
                diffY_data = np.subtract(gt_data, predY_data)

                predY_folder = "./results/"+test_para["test_para_name"]+"/predY/"
                diffY_folder = "./results/"+test_para["test_para_name"]+"/diffY/"
                if not os.path.exists(predY_folder):
                    os.makedirs(predY_folder)
                if not os.path.exists(diffY_folder):
                    os.makedirs(diffY_folder)

                predY_file = nibabel.Nifti1Image(predY_data, testX_file.affine, testX_file.header)
                diffY_file = nibabel.Nifti1Image(diffY_data, testX_file.affine, testX_file.header)
                predY_name = predY_folder+testX_name
                diffY_name = diffY_folder+testX_name
                nibabel.save(predY_file, predY_name)
                nibabel.save(diffY_file, diffY_name)

        save_folder = "./results/"+test_para["test_para_name"]
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        with open("./results/"+test_para["test_para_name"]+"/test_para_"+test_para["test_para_name"]+".json", "w") as outfile:  
            json.dump(test_para, outfile)
        with open("./results/"+test_para["test_para_name"]+"/train_para_"+test_para["train_para_name"]+".json", "w") as outfile:  
            json.dump(train_para, outfile)
        with open("./json/"+"/test_para_"+test_para["test_para_name"]+".json", "w") as outfile:  
            json.dump(test_para, outfile)
        
        # progress_folder = "./results/"+test_para["test_para_name"]+"/jpeg/"
        # if not os.path.exists(progress_folder):
        #     os.makedirs(progress_folder)
        # move_progress_cmd = "cp ./jpeg/"+test_para["train_para_name"]+"/*.jpg "+progress_folder
        # os.system(move_progress_cmd)

def create_index(dataA, n_slice, zeroPadding=False):
    h, w, z = dataA.shape
    index = np.zeros((z,n_slice))
    
    for idx_z in range(z):
        for idx_c in range(n_slice):
            index[idx_z, idx_c] = idx_z-(n_slice-idx_c+1)+n_slice//2+2
    if zeroPadding:
        index[index<0]=z
        index[index>z-1]=z
    else:
        index[index<0]=0
        index[index>z-1]=z-1
    return index

def createInput(data, n_slice=1):
    h, w, z = data.shape
    data_input = np.zeros((z, h, w, n_slice))
    index = create_index(data, n_slice, zeroPadding=False)
        
    for idx_z in range(z):
        for idx_c in range(n_slice):
            data_input[idx_z, :, :, idx_c] = data[:, :, int(index[idx_z, idx_c])]
            
    return data_input

if __name__ == '__main__':
    eval()