import argparse
import os
import time
import datetime
import dateutil.tz
import numpy as np
import cv2
import torch
import torch.autograd as autograd
from torch.autograd import Variable

# LIDC-IDRI-0011_7
# LIDC-IDRI-0012_1
# LIDC-IDRI-0016_1
# LIDC-IDRI-0081_1
# LIDC-IDRI-0150_1

vol_filename = 'LIDC-IDRI-0150_1.nii'
bg_filename = 'LIDC-IDRI-0012_1.nii'

vol_path = '/raid/LSTM_Synthesis_dataset/raw/vol/' + vol_filename
bg_path = '/raid/LSTM_Synthesis_dataset/raw/bg/' + bg_filename
mask_path = '/raid/LSTM_Synthesis_dataset/raw/mask/'

netG_path = '/home/sojeong/LSTM_Synthesis2/experiments/2020_12_04_15_02_46_0th_fold_nodule100/model/netG_final.pth'
netD_path = '/home/sojeong/LSTM_Synthesis2/experiments/2020_12_04_15_02_46_0th_fold_nodule100/model/netD_final.pth'

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu_id", type=str, default='0', help='GPU number')
    parser.add_argument("--num_workers", type=int, default=4, help='worker number')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    
    parser.add_argument("--num_sequence", type=int, default=3, help='number of sequence')
    parser.add_argument("--image_size", type=int, default=64, help='image size')
    parser.add_argument("--feature_dim", type=int, default=12, help='feature dimension')

    parser.add_argument('--nodule_lambda', type=float, default=100.0, help='parameter for nodule part')
    parser.add_argument('--background_lambda', type=float, default=50.0, help='parameter for background part')
    parser.add_argument('--feature_lambda', type=float, default=0.1, help='batch size')
    parser.add_argument('--yn_lambda', type=float, default=1.0, help='batch size')

    parser.add_argument("--data_dir", type=str, default='/raid/LSTM_Synthesis_dataset/sequence/', help='dataset path')
    # parser.add_argument("--exp_name", type=str, default='test1', required=True, help='experiment name')

    opt = parser.parse_args()
    print(opt)



    from dataset import NoduleDataset
    nodule_dataset = NoduleDataset(vol_path, bg_path)
    dataloader = torch.utils.data.DataLoader(nodule_dataset, batch_size = 32, drop_last=False, shuffle=False, num_workers=4)

    device = torch.device("cuda:%s" % opt.gpu_id)
    from model import DeepSequentialNet, Discriminator
    netG = DeepSequentialNet(num_sequence=opt.num_sequence, feature_dim=opt.feature_dim, device=device).to(device)
    netD = Discriminator(feature_dim=opt.feature_dim).to(device)

    netG.load_state_dict(torch.load(netG_path))
    netD.load_state_dict(torch.load(netD_path))

    netG.train()
    netG.requires_grad_(False)
    netD.train()
    netD.requires_grad_(False)

    for data in dataloader:
        masked_vol_sequence, bg_sequence, feature_sequence, real_slice, nodule_mask, bg_mask = data
        masked_vol_sequence = Variable(masked_vol_sequence).float().to(device)
        bg_sequence = Variable(bg_sequence).float().to(device)
        feature_sequence = Variable(feature_sequence).float().to(device)
        real_slice = Variable(real_slice).float().to(device)
        nodule_mask = Variable(nodule_mask).float().to(device)
        bg_mask = Variable(bg_mask).float().to(device)

        with torch.no_grad():
            ### prediction
            synth_from_vol = netG(masked_vol_sequence, feature_sequence)
            synth_from_bg = netG(bg_sequence, feature_sequence)

            ### gpu to cpu
            synth_from_vol = synth_from_vol.cpu().numpy()
            synth_from_vol = np.squeeze(synth_from_vol)

            synth_from_bg = synth_from_bg.cpu().numpy()
            synth_from_bg = np.squeeze(synth_from_bg)

            real_slice = real_slice.cpu().numpy()
            real_slice = np.squeeze(real_slice)

            bg_sequence = bg_sequence[:,1,:,:,:].cpu().numpy()
            bg_sequence = np.squeeze(bg_sequence)

            masked_vol_sequence = masked_vol_sequence.cpu().numpy()
            masked_vol_sequence = np.transpose(masked_vol_sequence, (0, 2, 3, 4, 1))   ###(b,c,x,y,s)
            masked_vol_sequence = masked_vol_sequence[:, 0, :, :, 1]

            nodule_mask = nodule_mask.cpu().numpy()
            nodule_mask = np.where(nodule_mask==1, 255, nodule_mask)
            nodule_mask = np.squeeze(nodule_mask)
            bg_mask = bg_mask.cpu().numpy()
            bg_mask = np.where(bg_mask==1, 255, bg_mask)
            bg_mask = np.squeeze(bg_mask)

            result_imgs = np.array([])
            for train_result_idx, (gt, seq1, pred1, seq2, pred2, nm, bgm) in enumerate(zip(real_slice, masked_vol_sequence, synth_from_vol, bg_sequence, synth_from_bg, nodule_mask, bg_mask)):
                gt = gt*255
                seq1 = seq1*255
                pred1 = pred1*255
                seq2 = seq2*255
                pred2 = pred2*255

                result_img = np.concatenate((gt, seq1, pred1, seq2, pred2, nm, bgm), 1)

                if train_result_idx == 0:
                    result_imgs = result_img
                else:
                    result_imgs = np.concatenate((result_imgs, result_img), 0)
            cv2.imwrite('samples/' + os.path.splitext(vol_filename)[0] + '_result2.png', result_imgs)







