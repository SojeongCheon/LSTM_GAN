from numpy.lib.shape_base import apply_along_axis
import torch.utils.data as data
import pandas as pd
import numpy as np
from medpy.io import load, save
import random
import os
import pandas as pd

NUM_SEQUENCE = 3

class Extractor:
    def __init__(self, data_dir):
        self.data_dir = data_dir 
        self.vol_filenames = self.load_filenames(self.data_dir + 'vol/')
        self.bg_filenames = self.load_filenames(self.data_dir + 'bg/')
        
        self.csv_info = self.get_info('/raid/LSTM_Synthesis_dataset/lidc_features.csv')

    def load_filenames(self, data_dir):
        filenames = os.listdir(data_dir)
        return filenames

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

    # def apply_zerobox(self, vol):
    #     temp = vol.copy()
    #     r = cfg.MASK_SIZE // 2
    #     c = cfg.VOLUME_SIZE // 2
    #     temp[c-r-1:c+r, c-r-1:c+r , : ] = 0
    #     return temp

    def extract(self, plot=True):
        num_seq = int((NUM_SEQUENCE - 1)/2)
        seq_vols = []
        seq_masks = []
        seq_info_list = []
        
        #### background
        for bg_filename in self.bg_filenames:
            bg_path = self.data_dir + 'bg/' + bg_filename
            bg = self.get_volume(bg_path, normalize = True)

            bg_depth = random.randrange(2, 40)
            bg_depth_min = 32 - bg_depth//2
            bg_depth_max = 32 + bg_depth//2
            if bg_depth%2 == 1:
                bg_depth_max = 32 + bg_depth//2 + 1

            bg_slice_cnt = 0
            for z in range(bg_depth_min, bg_depth_max):
                seq_bg_filename = '/raid/LSTM_Synthesis_dataset/sequence/bg_sequence/' + os.path.splitext(bg_filename)[0] + '_z' + str(bg_slice_cnt) + "_d" + str(bg_depth) + '.nii'
                seq_bg = bg[:,:, z-num_seq:z+1+num_seq]
                bg_slice_cnt += 1
                save(seq_bg, seq_bg_filename)



        # ### volume, mask
        # for filename in self.vol_filenames:
        #     scan_path = self.data_dir + 'vol/' + filename
        #     mask_path = self.data_dir + 'mask/' + filename
        #     vol = self.get_volume(scan_path, normalize = True)
        #     mask = self.get_volume(mask_path, normalize = False)

        #     cnt = 0
            
        #     for z in range(64):
        #         if np.max(mask[:,:,z]) == 0.0:
        #             continue
        #         seq_vol_filename = '/raid/LSTM_Synthesis_dataset/' + self.mode + '/vol_slice/' + os.path.splitext(filename)[0] + '_z' + str(cnt) + '.nii'
        #         seq_mask_filename = '/raid/LSTM_Synthesis_dataset/' + self.mode + '/mask/' + os.path.splitext(filename)[0] + '_z' + str(cnt) + '.nii'
        #         seq_vol_zerobox_filename = '/raid/LSTM_Synthesis_dataset/' + self.mode + '/vol_sequence/' + os.path.splitext(filename)[0] + '_z' + str(cnt) + '.nii'
        #         output_vol = vol[:,:,z]
        #         seq_mask = mask[:,:, z-num_seq:z+1+num_seq]
        #         seq_vol = vol[:,:, z-num_seq:z+1+num_seq]

        #         # save(output_vol, seq_vol_filename)                       ### output
        #         # save(seq_mask, seq_mask_filename)                        ### useless
        #         # save(seq_vol, seq_vol_zerobox_filename)                  ### input

        #         # seq_vols.append(seq_vol)
        #         # seq_masks.append(seq_mask)

        #         cnt += 1
            
        #     temp_info = self.find_info(filename)
         
        #     seq_info = []
        #     seq_info.append(temp_info[0])      # patient_id
        #     seq_info.append(temp_info[1])      # nodule_id
        #     seq_info.append(cnt)               # total slice number
        #     seq_info.append(temp_info[2])      # diameter
        #     seq_info.append(temp_info[3])      # subtlety     
        #     seq_info.append(temp_info[4])      # internalStructure
        #     seq_info.append(temp_info[5])      # calcification
        #     seq_info.append(temp_info[6])      # sphericity
        #     seq_info.append(temp_info[7])      # margin
        #     seq_info.append(temp_info[8])      # lobulation
        #     seq_info.append(temp_info[9])      # spiculation
        #     seq_info.append(temp_info[10])     # texture
        #     seq_info.append(temp_info[11])     # malignancy
        #     seq_info_list.append(seq_info)
            
        
        # csv_save_path = '/raid/LSTM_Synthesis_dataset/' + self.mode + '/seq_info2.csv'
        # data_header = ['patient_id', 'nodule_id', 'total_slice', 'diameter', 'subtlety', 'internalStructure', 'calcification', 'sphericity', 'margin', 'lobulation', 'spiculation', 'texture', 'malignancy']

        # dataframe = pd.DataFrame(seq_info_list)
        # dataframe.to_csv(csv_save_path, header = data_header, index=False)
        

        # seq_vols = np.array(seq_vols)
        # seq_masks = np.array(seq_masks)
        # print(seq_vols.shape)
        # print(seq_masks.shape)
            
if __name__ == "__main__":
    builder = Extractor('/raid/LSTM_Synthesis_dataset/raw/')
    builder.extract()
            
