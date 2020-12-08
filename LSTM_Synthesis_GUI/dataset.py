import torch.utils.data as data
import pandas as pd
import numpy as np
from medpy.io import load, save
import random
import cv2
import os
import pandas as pd

NUM_SEQUENCE = 3

class NoduleDataset(data.Dataset):
    def __init__(self, vol_path, bg_path):
        self.filename = os.path.splitext(os.path.basename(vol_path))[0]
        self.vol = self.get_volume(vol_path, True)
        self.bg = self.get_volume(bg_path, True)
        self.mask = self.get_volume('/raid/LSTM_Synthesis_dataset/raw/mask/' + os.path.basename(vol_path), False)
        self.csv_info = self.get_info('/raid/LSTM_Synthesis_dataset/lidc_features.csv')

        self.seq_vols, self.seq_bgs, self.seq_masks, self.seq_features = self.get_seq_dataset()

    def get_volume(self, path, normalize = True):
        volume, _ = load(path)
    
        if normalize:
            volume = (volume - np.min(volume)) / (np.max(volume) - np.min(volume))
        return volume
    
    def get_info(self, data_dir):
        features_csv = pd.read_csv(data_dir)
        features = np.array([features_csv.patient_id,             ### 0
                            features_csv.nodule_id,               ### 1
                            features_csv.diameter,                ### 2
                            features_csv.subtlety,                ### 3
                            features_csv.internalStructure,       ### 4
                            features_csv.calcification,           ### 5
                            features_csv.sphericity,              ### 6
                            features_csv.margin,                  ### 7
                            features_csv.lobulation,              ### 8
                            features_csv.spiculation,             ### 9
                            features_csv.texture,                 ### 10
                            features_csv.malignancy,              ### 11
                            ])
        features = np.transpose(features)
        return features
    
    def find_info(self, filename):
        for info in self.csv_info:
            csv_info_filename = info[0] + '_' + str(info[1])
            if os.path.splitext(filename)[0] == csv_info_filename:
                return info
        return

    def get_seq_dataset(self):
        num_seq = int((NUM_SEQUENCE - 1)/2)
        seq_vols = []         # ***********************
        seq_bgs = []          # ***********************
        seq_masks = []        # ***********************
        seq_features = []     # ***********************

        ### seq_vols, seq_masks
        total_slice = 0
        for z in range(64):
            if np.max(self.mask[:,:,z]) == 0.0:
                continue

            seq_vols.append(self.vol[:,:, z-num_seq:z+1+num_seq])
            seq_bgs.append(self.bg[:,:, z-num_seq:z+1+num_seq])
            seq_masks.append(self.mask[:,:, z-num_seq:z+1+num_seq])
            total_slice += 1
        
        ### seq_features
        info = self.find_info(self.filename)
        for s in range(total_slice):
            temp_feature = []
            temp_feature.append(s)
            temp_feature.append(total_slice)
            temp_feature.append(info[2])
            temp_feature.append(info[3])
            temp_feature.append(info[4])
            temp_feature.append(info[5])
            temp_feature.append(info[6])
            temp_feature.append(info[7])
            temp_feature.append(info[8])
            temp_feature.append(info[9])
            temp_feature.append(info[10])
            temp_feature.append(info[11])
            seq_features.append(temp_feature)
        
        seq_features = np.array(seq_features)
        return seq_vols, seq_bgs, seq_masks, seq_features


    def __len__(self):
        return len(self.seq_vols)

    def __getitem__(self, index):
        ### mask    
        nodule_mask = self.seq_masks[index]                                                      ### (x, y, seq_slices)
        
        nodule_mask = nodule_mask[:,:,1]
        temp_bg_mask = 1.0 - nodule_mask
        bg_mask = temp_bg_mask.copy()
        ####################################################
        nodule_mask = np.expand_dims(nodule_mask, axis = 0)
        bg_mask = np.expand_dims(bg_mask, axis = 0)
        
        temp_bg_mask = np.expand_dims(temp_bg_mask, axis = 2)
        temp_bg_mask = np.concatenate((temp_bg_mask, temp_bg_mask, temp_bg_mask), 2)
        
        ### input slice sequence, masked input slice sequnce   (x, x')
        vol_sequence =  self.seq_vols[index]                                          ### (x, y, seq_slices)
        
        gt_slice = vol_sequence[:,:,1].copy()
        gt_slice = np.expand_dims(gt_slice, axis = 0)                                   ### (channel, x, y)

        masked_vol_sequence = temp_bg_mask * vol_sequence
        masked_vol_sequence = np.expand_dims(masked_vol_sequence, axis=0)               ### (channel, x, y, seq_slices)
        masked_vol_sequence = np.transpose(masked_vol_sequence, (3,0,1,2))              ### (seq_slices, channel, x, y)     
 
        ### background
        bg_sequence =  self.seq_bgs[index]
        ###### version 1
        # bg_sequence = np.expand_dims(bg_sequence, axis=0)                               ### (channel, x, y, seq_slices)
        # bg_sequence = np.transpose(bg_sequence, (3,0,1,2))                              ### (seq_slices, channel, x, y)     
        ###### version 2
        masked_bg_sequence = temp_bg_mask * bg_sequence
        masked_bg_sequence = np.expand_dims(masked_bg_sequence, axis=0)                               ### (channel, x, y, seq_slices)
        masked_bg_sequence = np.transpose(masked_bg_sequence, (3,0,1,2))                              ### (seq_slices, channel, x, y)
       
        ### feature information
        feature_sequence = self.seq_features[index]
        
        return masked_vol_sequence, masked_bg_sequence, feature_sequence, gt_slice, nodule_mask, bg_mask

    



            
