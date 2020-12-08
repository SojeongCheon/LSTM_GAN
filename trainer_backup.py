#####################################################################
# WGAN-gp
#####################################################################

from os import PRIO_PGRP
from config import cfg

import os
import time
import torch
import torch.nn as nn
import torch.autograd as autograd
from torch.autograd import Variable
import torch.optim as optim
import torch.backends.cudnn as cudnn

from torch.utils.tensorboard import SummaryWriter
writer_path = cfg.EXP_PATH + '/runs'
os.makedirs(writer_path)
writer = SummaryWriter(writer_path)

import numpy as np
import random
import torchvision
import matplotlib.pyplot as plt
from medpy.io import load, save
import cv2

from model import DeepSequentialNet, Discriminator

LAMBDA = 10
CRITIC_ITERS = 5

def weights_init(m):
    classname = m.__class__.__name__
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_uniform_(m.weight.data)
    if isinstance(m, nn.BatchNorm2d):
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def calc_gradient_penalty(netD, real_data, fake_data, device):
    alpha = torch.rand(real_data.size()[0], 1, 1, 1)
    alpha = alpha.expand(real_data.size())
    alpha = alpha.to(device)

    interpolates = alpha * real_data + ((1 - alpha) * fake_data)
    interpolates = interpolates.to(device)
    interpolates = autograd.Variable(interpolates, requires_grad=True)

    disc_interpolates,_,_ = netD(interpolates)

    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size()).to(device),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]

    gradients = gradients.view(gradients.size(0), -1) + 1e-16
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * LAMBDA
    return gradient_penalty

class sequentialSynthesisTrainer(object):
    def __init__(self, epochs, gpu, batch_size, d_learning_rate, g_learning_rate, nodule_lambda, background_lambda, feature_lambda, yn_lambda, output_dir, train_dataloader, test_dataloader):
        self.model_dir = os.path.join(output_dir, 'model')
        self.train_result_dir = os.path.join(output_dir, 'result', 'train')
        self.test_result_dir = os.path.join(output_dir, 'result', 'test')
        os.makedirs(self.model_dir)
        os.makedirs(self.train_result_dir)
        os.makedirs(self.test_result_dir)

        self.epochs = epochs
        self.gpu = gpu
        self.device = torch.device("cuda:%s" % self.gpu)
       
        self.batch_size = batch_size
        self.d_learning_rate = d_learning_rate
        self.g_learning_rate = g_learning_rate
        self.nodule_lambda = nodule_lambda
        self.background_lambda = background_lambda
        # self.feature_lambda = feature_lambda
        # self.yn_lambda = yn_lambda
        self.output_dir = output_dir

        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        
    def train(self):
        netG = DeepSequentialNet().to(self.device)
        netG.apply(weights_init)
        netD = Discriminator().to(self.device)
        netD.apply(weights_init)

        print("********************************************netG********************************************")
        print(netG)
        print("********************************************netD********************************************")
        print(netD)
        print("********************************************************************************************")
        print("********************************************************************************************")
        print("********************************************************************************************")

        ### optimizer, loss func
        optimizerG = optim.Adam(netG.parameters(), lr=self.g_learning_rate, betas=(0.5, 0.999))
        optimizerD = optim.Adam(netD.parameters(), lr=self.d_learning_rate, betas=(0.5, 0.999))
        
        MSE_criterion = nn.MSELoss().to(self.device)
        L1_criterion = nn.L1Loss().to(self.device)

        # real, fake label
        real_labels = Variable(torch.FloatTensor(self.batch_size).fill_(1)).to(self.device)
        fake_labels = Variable(torch.FloatTensor(self.batch_size).fill_(0)).to(self.device)

        total_step = 0
        for epoch in range(self.epochs):
            netG.train()
            netD.train()

            for data in self.train_dataloader:
            
                start_t = time.time()
                
                ### data preparation
                masked_vol_sequence, bg_sequence, feature_sequence, real_slice, nodule_mask, bg_mask = data
                masked_vol_sequence = Variable(masked_vol_sequence).float().to(self.device)
                bg_sequence = Variable(bg_sequence).float().to(self.device)
                feature_sequence = Variable(feature_sequence).float().to(self.device)
                real_slice = Variable(real_slice).float().to(self.device)
                nodule_mask = Variable(nodule_mask).float().to(self.device)
                bg_mask = Variable(bg_mask).float().to(self.device)

                ### generate fake one
                netG.requires_grad_(True)
                fake_slice1 = netG(masked_vol_sequence, feature_sequence)
                fake_slice2 = netG(bg_sequence, feature_sequence)

                ##########################
                ### update discriminator
                ##########################
                netD.requires_grad_(True)
                real_logit1, real_feature1, real_yn1 = netD(real_slice)       
                fake_logit1, fake_feature1, fake_yn1 = netD(fake_slice1.detach())
                fake_logit2, fake_feature2, fake_yn2 = netD(fake_slice2.detach())
                gradient_penalty1 = calc_gradient_penalty(netD, real_slice, fake_slice1.detach(), self.device)


                errD_wgan_gp1 = -torch.mean(real_logit1) + torch.mean(fake_logit1) + gradient_penalty1 
                errD_feature = MSE_criterion(real_feature1, feature_sequence) + MSE_criterion(fake_feature1, feature_sequence) + MSE_criterion(fake_feature2, feature_sequence)
                errD_yn = MSE_criterion(real_yn1, real_labels) + MSE_criterion(fake_yn1, real_labels) + MSE_criterion(fake_yn2, real_labels)

                errD = errD_wgan_gp1 + errD_feature + errD_yn

                netD.zero_grad()
                errD.backward()
                optimizerD.step()
                netD.requires_grad_(False)
                    
                ##########################
                ### update generator
                ##########################
                errG = 0
                errG_nodule = 0
                errG_bg = 0
                errG_feature = 0
                errG_yn = 0
                if total_step % CRITIC_ITERS == 0:
                    pred_logit1, pred_feature1, pred_yn1 = netD(fake_slice1)
                    pred_logit2, pred_feature2, pred_yn2 = netD(fake_slice2)
                    
                    errG_nodule = L1_criterion(fake_slice1*nodule_mask, real_slice*nodule_mask) + L1_criterion(fake_slice2*nodule_mask, real_slice*nodule_mask) 
                    errG_bg = L1_criterion(fake_slice1*bg_mask, real_slice*bg_mask) + L1_criterion(fake_slice2*bg_mask, bg_sequence[:,1,:,:,:]*bg_mask) 
                    errG_feature = MSE_criterion(pred_feature1, feature_sequence) + MSE_criterion(pred_feature2, feature_sequence)
                    errG_yn = MSE_criterion(pred_yn1, real_labels) + MSE_criterion(pred_yn2, real_labels) 

                    errG = - torch.mean(pred_logit1) - torch.mean(pred_logit2) + self.nodule_lambda * errG_nodule + self.background_lambda * errG_bg + errG_feature + errG_yn

                    netG.zero_grad()
                    errG.backward()
                    optimizerG.step()
                    netG.requires_grad_(False)
            
                if total_step % 100 == 0:
                    end_t = time.time()  
                    
                    ### print loss
                    print('[%d / %d - %d step] errD : %.5f  errG : %.5f  err_nodules : %.5f err_backgrounds : %.5f time : %.2fs' % (epoch, self.epochs, total_step, errD, errG, errG_nodule, errG_bg, end_t - start_t))                
                    
                    writer.add_scalar('errD', errD, total_step)
                    writer.add_scalar('errD_wgan_gp1', errD_wgan_gp1, total_step)
                    writer.add_scalar('errD_feature', errD_feature, total_step)
                    writer.add_scalar('errD_yn', errD_yn, total_step)


                    writer.add_scalar('errG', errG, total_step)
                    writer.add_scalar('err_nodules', errG_nodule, total_step)
                    writer.add_scalar('err_backgrounds', errG_bg, total_step)
                    writer.add_scalar('errG_feature', errG_feature, total_step)
                    writer.add_scalar('errG_yn', errG_yn, total_step)

                    ## print result image
                    netG.eval()
                    netD.eval()
                    netG.requires_grad_(False)
                    netD.requires_grad_(False)
                    with torch.no_grad():
                        fake_slice1 = fake_slice1.cpu().numpy()
                        fake_slice1 = np.squeeze(fake_slice1)

                        fake_slice2 = fake_slice2.cpu().numpy()
                        fake_slice2 = np.squeeze(fake_slice2)

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
                        for train_result_idx, (gt, seq1, pred1, seq2, pred2, nm, bgm) in enumerate(zip(real_slice, masked_vol_sequence, fake_slice1, bg_sequence, fake_slice2, nodule_mask, bg_mask)):
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
                        
                            if train_result_idx==6:
                                break
                        cv2.imwrite(self.train_result_dir + '/epoch_' + str(epoch) + '_step' + str(total_step) +'.png', result_imgs)
                    
                    netG.train()
                    netD.train()
                        
                total_step += 1
            ##########################################################
            ##########################################################
            ##########################################################
            netG.eval()
            netD.eval()
            netG.requires_grad_(False)
            netD.requires_grad_(False)
            
            with torch.no_grad():
                for data in self.test_dataloader:
                    test_masked_vol_sequence, test_bg_sequence, test_feature_sequence, test_real_slice, nodule_mask, bg_mask = data
                    test_masked_vol_sequence = Variable(test_masked_vol_sequence).float().to(self.device)
                    test_bg_sequence = Variable(test_bg_sequence).float().to(self.device)
                    test_feature_sequence = Variable(test_feature_sequence).float().to(self.device)
                    test_real_slice = Variable(test_real_slice).float().to(self.device)
                    nodule_mask = Variable(nodule_mask).float().to(self.device)
                    bg_mask = Variable(bg_mask).float().to(self.device)
                    
                    test_fake_slice1 = netG(test_masked_vol_sequence, test_feature_sequence)
                    test_fake_slice2 = netG(test_bg_sequence, test_feature_sequence)

                    #######################################################
                    #######################################################  
                    fake_logit1, fake_feature1, fake_yn1 = netD(test_fake_slice1.detach())
                    fake_logit2, fake_feature2, fake_yn2 = netD(test_fake_slice2.detach())
                    
                    test_errG_nodule = L1_criterion(test_fake_slice1*nodule_mask, test_real_slice*nodule_mask) + L1_criterion(test_fake_slice2*nodule_mask, test_real_slice*nodule_mask) 
                    test_errG_bg = L1_criterion(test_fake_slice1*bg_mask, test_real_slice*bg_mask) + L1_criterion(test_fake_slice2*bg_mask, test_bg_sequence[:,1,:,:,:]*bg_mask) 
                    test_errG_feature = MSE_criterion(fake_feature1, test_feature_sequence) + MSE_criterion(fake_feature2, test_feature_sequence)
                    test_errG_yn = MSE_criterion(fake_yn1, real_labels) + MSE_criterion(fake_yn2, real_labels) 

                    test_errG = - torch.mean(fake_logit1) - torch.mean(fake_logit2) + self.nodule_lambda * test_errG_nodule + self.background_lambda * test_errG_bg + test_errG_feature + test_errG_yn
                    
                    print('[%d / %d - %d step] errG : %.5f  err_nodules : %.5f err_backgrounds : %.5f ' % (epoch, self.epochs, total_step, test_errG, test_errG_nodule, test_errG_bg))
                    writer.add_scalar('test_errG', test_errG, total_step)
                    writer.add_scalar('test_err_nodule', test_errG_nodule, total_step)
                    writer.add_scalar('test_err_background', test_errG_bg, total_step)
                    writer.add_scalar('test_err_background', test_errG_feature, total_step)
                    writer.add_scalar('test_err_background', test_errG_yn, total_step)

                    #######################################################
                    #######################################################
                    test_fake_slice1 = test_fake_slice1.cpu().numpy()
                    test_fake_slice1 = np.squeeze(test_fake_slice1)

                    test_fake_slice2 = test_fake_slice2.cpu().numpy()
                    test_fake_slice2 = np.squeeze(test_fake_slice2)

                    test_real_slice = test_real_slice.cpu().numpy()
                    test_real_slice = np.squeeze(test_real_slice)

                    test_bg_sequence = test_bg_sequence.cpu().numpy()
                    test_bg_sequence = np.transpose(test_bg_sequence, (0, 2, 3, 4, 1))                   ###(b,c,x,y,s)
                    test_bg_sequence = test_bg_sequence[:, 0, :, :, 1]

                    test_masked_vol_sequence = test_masked_vol_sequence.cpu().numpy()
                    test_masked_vol_sequence = np.transpose(test_masked_vol_sequence, (0, 2, 3, 4, 1))   ###(b,c,x,y,s)
                    test_masked_vol_sequence = test_masked_vol_sequence[:, 0, :, :, 1]

                    nodule_mask = nodule_mask.cpu().numpy()
                    nodule_mask = np.where(nodule_mask==1, 255, nodule_mask)
                    nodule_mask = np.squeeze(nodule_mask)
                    bg_mask = bg_mask.cpu().numpy()
                    bg_mask = np.where(bg_mask==1, 255, bg_mask)
                    bg_mask = np.squeeze(bg_mask)

                    result_imgs = np.array([])
                    for idx, (gt, seq1, pred1, seq2, pred2, nm, bgm) in enumerate(zip(test_real_slice, test_masked_vol_sequence, test_fake_slice1, test_bg_sequence, test_fake_slice2, nodule_mask, bg_mask)):
                        gt = gt*255
                        seq1 = seq1*255
                        pred1 = pred1*255
                        seq2 = seq2*255
                        pred2 = pred2*255

                        result_img = np.concatenate((gt, seq1, pred1, seq2, pred2, nm, bgm), 1)


                        if idx == 0:
                            result_imgs = result_img
                        else:
                            result_imgs = np.concatenate((result_imgs, result_img), 0)
                    
                        if idx==6:
                            break
                    cv2.imwrite(self.test_result_dir + '/epoch_' + str(epoch) + '_step' + str(total_step) +'.png', result_imgs)
                    break  

                if epoch % 5 == 0:
                    torch.save(netG.state_dict(), '%s/netG_epoch%d.pth' % (self.model_dir, epoch))
                    torch.save(netD.state_dict(), '%s/netD_epoch%d.pth' % (self.model_dir, epoch))
        torch.save(netG.state_dict(), '%s/netG_final.pth' % (self.model_dir))
        torch.save(netD.state_dict(), '%s/netD_final.pth' % (self.model_dir))


            
            