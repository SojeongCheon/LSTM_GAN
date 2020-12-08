import os
import random
import argparse
import time
import datetime
import dateutil.tz
import torch

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu_id", type=str, default='1', help='GPU number')
    parser.add_argument("--num_workers", type=int, default=4, help='worker number')
    parser.add_argument("--epochs", type=int, default=300, help="epochs")
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--generator_lr', type=float, default=1e-4, help='generator learning rate')
    parser.add_argument('--discriminator_lr', type=float, default=1e-4, help='discriminator learning rate')
    parser.add_argument('--total_fold_num', type=int, default=10, help='num of total fold')
    parser.add_argument('--fold_num', type=int, nargs='+', required=True, help='num of target fold')
    
    parser.add_argument("--num_sequence", type=int, default=3, help='number of sequence')
    parser.add_argument("--image_size", type=int, default=64, help='image size')
    parser.add_argument("--feature_dim", type=int, default=12, help='feature dimension')

    parser.add_argument('--nodule_lambda', type=float, default=100.0, help='parameter for nodule part')
    parser.add_argument('--background_lambda', type=float, default=50.0, help='parameter for background part')
    parser.add_argument('--feature_lambda', type=float, default=0.1, help='batch size')
    parser.add_argument('--yn_lambda', type=float, default=1.0, help='batch size')

    parser.add_argument("--data_dir", type=str, default='/raid/LSTM_Synthesis_dataset/sequence/', help='dataset path')
    parser.add_argument("--exp_name", type=str, default='test1', required=True, help='experiment name')

    opt = parser.parse_args()
    print(opt)
    


    from dataset_fold import NoduleDataset
    train_dataset = []
    test_dataset =[]
    for fold in range(opt.total_fold_num):
        train_dataset.append(NoduleDataset(opt.data_dir,'train', fold, opt.total_fold_num))
        test_dataset.append(NoduleDataset(opt.data_dir,'test', fold, opt.total_fold_num))

    for fold in opt.fold_num:
        opt.exp_name = str(fold) + 'th_fold_' + opt.exp_name

        now = datetime.datetime.now(dateutil.tz.tzlocal())
        timestamp = now.strftime('%Y_%m_%d_%H_%M')
        output_dir = './experiments/%s_%s' % (timestamp, opt.exp_name)


        print(fold, '-th fold ::: training start')
        train_dataloader = torch.utils.data.DataLoader(train_dataset[fold], batch_size=opt.batch_size, drop_last=True, shuffle=True, num_workers=opt.num_workers)
        test_dataloader = torch.utils.data.DataLoader(test_dataset[fold], batch_size=opt.batch_size, shuffle=False, num_workers=opt.num_workers)
        
        from torch.utils.tensorboard import SummaryWriter
        writer_path = 'log/SequentialSynthetic_%s_%s' % (timestamp, opt.exp_name)
        os.makedirs(writer_path)
        writer = SummaryWriter(writer_path)

        from trainer import sequentialSynthesisTrainer as trainer
        algo = trainer(writer, opt.epochs, opt.gpu_id, opt.batch_size, opt.discriminator_lr, opt.generator_lr, opt.num_sequence, opt.image_size, opt.feature_dim, opt.nodule_lambda, opt.background_lambda, opt.feature_lambda, opt.yn_lambda, output_dir, train_dataloader, test_dataloader)
    
        start_t = time.time()
        algo.train()
        end_t = time.time()

        print(fold, '-th fold ::: total time for training: ', end_t - start_t)
  

