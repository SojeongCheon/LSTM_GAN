import torch.utils.data as data
import numpy as np
import os
from medpy.io import load
import pandas as pd
import random
import cv2


class NoduleDataset(data.Dataset):
    def __init__(self, data_dir, mode):
        self.data_dir = data_dir + mode
        self.vol_filenames = self.load_filenames(self.data_dir + '/vol_slice/')
        self.bg_filenames = self.load_filenames(self.data_dir + '/bg_sequence/')
        self.info = self.load_info(self.data_dir + '/seq_info2.csv')
       
    def load_info(self, data_dir):
        csv = pd.read_csv(data_dir)
        info = np.array([csv.patient_id,       ### 0
                        csv.nodule_id,         ### 1
                        csv.total_slice,       ### 2
                        csv.diameter,          ### 3
                        csv.subtlety,          ### 4
                        csv.internalStructure, ### 5
                        csv.calcification,     ### 6
                        csv.sphericity,        ### 7
                        csv.margin,            ### 8
                        csv.lobulation,        ### 9
                        csv.spiculation,       ### 10
                        csv.texture,           ### 11
                        csv.malignancy])       ### 12
        info = np.transpose(info)
        return info

    def load_filenames(self, data_dir):
        filenames = os.listdir(data_dir)
        return filenames
    
    def find_info(self, filename):
        current_seq_filename = filename.split('_')[0] + '_' + filename.split('_')[1]          ### current sequence name
        slice_num = filename.split('_')[-1].split('.')[0].replace("z","")

        for i, data in enumerate(self.info):
            csv_filename = data[0] + '_' + str(data[1])      ### filename in csv
            if csv_filename == current_seq_filename:
                # feature = np.array([int(slice_num), data[2],data[3]])
                feature = np.array([int(slice_num), data[2],data[3],data[4],data[5],data[6],data[7],data[8],data[9],data[10],data[11],data[12]])
                return feature         ### total_slice, diameter, .... , malignancy   >>>  12 features
        return 0

    def __len__(self):
        return len(self.vol_filenames)

    def __getitem__(self, index):
        ### mask     (m)
        mask_path = self.data_dir + '/mask/' + self.vol_filenames[index]  
        nodule_mask, _ = load(mask_path)                                                      ### (x, y, seq_slices)
        
        nodule_mask = nodule_mask[:,:,1]
        temp_bg_mask = 1.0 - nodule_mask
        ####################################################  ###############################################################################################################################################################################
        # kernel = np.ones((7, 7), np.uint8)
        # bg_mask = cv2.erode(temp_bg_mask, kernel, iterations=1)
        bg_mask = temp_bg_mask.copy()
        ####################################################
        nodule_mask = np.expand_dims(nodule_mask, axis = 0)
        bg_mask = np.expand_dims(bg_mask, axis = 0)
        
        temp_bg_mask = np.expand_dims(temp_bg_mask, axis = 2)
        temp_bg_mask = np.concatenate((temp_bg_mask, temp_bg_mask, temp_bg_mask), 2)
        
        ### input slice sequence, masked input slice sequnce   (x, x')
        vol_sequence_path = self.data_dir + '/vol_sequence/' + self.vol_filenames[index]
        vol_sequence, _ = load(vol_sequence_path)                                       ### (x, y, seq_slices)
        
        gt_slice = vol_sequence[:,:,1].copy()
        gt_slice = np.expand_dims(gt_slice, axis = 0)                                   ### (channel, x, y)

        masked_vol_sequence = temp_bg_mask * vol_sequence
        masked_vol_sequence = np.expand_dims(masked_vol_sequence, axis=0)               ### (channel, x, y, seq_slices)
        masked_vol_sequence = np.transpose(masked_vol_sequence, (3,0,1,2))              ### (seq_slices, channel, x, y)     
 
        ### background
        bg_index = random.randrange(0, len(self.bg_filenames))
        bg_sequence_path = self.data_dir + '/bg_sequence/' + self.bg_filenames[bg_index]
        bg_sequence, _ = load(bg_sequence_path)
        bg_sequence = np.expand_dims(bg_sequence, axis=0)                               ### (channel, x, y, seq_slices)
        bg_sequence = np.transpose(bg_sequence, (3,0,1,2))                              ### (seq_slices, channel, x, y)     

        ### feature information
        feature_sequence = self.find_info(self.vol_filenames[index])
        
        return masked_vol_sequence, bg_sequence, feature_sequence, gt_slice, nodule_mask, bg_mask



