#from __future__ import print_function
#torch
import time,random,os,sys
os.environ['CUDA_VISIBLE_DEVICES'] = "1"
import torch
import torch.nn as nn
import torch.optim as optim
import logging
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torch.nn.utils as utils
import torchaudio
#sys
import math
import numpy as np
from random import shuffle
#import matplotlib.pyplot as plot
import argparse
import matplotlib.pyplot as plt
import pandas as pd
#my
from torch.utils.data import Dataset, DataLoader
import warnings
# from torch.multiprocessing import set_start_method
# set_start_method('spawn')
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

###  totally reproduce
def setup_seed(seed=1):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

setup_seed(30)

# from load_data import MyDataset
# Training settings
parser = argparse.ArgumentParser(description='PyTorch Speaker Recognition')
# data path

parser.add_argument('--log-dir', default='./log/',
                    help='folder to output model checkpoints')
parser.add_argument('--resume',
                    default=None,
                    type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--start-epoch', default=1, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--epochs', type=int, default=40, metavar='E',
                    help='number of epochs to train (default: 10)')
# Training options
parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                    help='learning rate (default: 0.125)')
parser.add_argument('--lr-decay', default=1e-5, type=float, metavar='LRD',
                    help='learning rate decay ratio (default: 1e-4')
parser.add_argument('--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 0.0)')
parser.add_argument('--optimizer', default='sgd', type=str,
                    metavar='OPT', help='The optimizer to use (default: Adagrad)')
# Device options
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=0, metavar='S',
                    help='random seed (default: 0)')
#cut mix
parser.add_argument('--is-training', type=int,
                    help='start from MFB file')
parser.add_argument('--fold', type=str,
                    help='fold1-6')
parser.add_argument('--log-interval', type=int, default=100, metavar='LI',
                    help='how many batches to wait before logging training status')
parser.add_argument('--init-clip-max-norm', default=1.0, type=float,
                    metavar='CLIP', help='grad clip max norm (default: 1.0)')
args = parser.parse_args()

# set the device to use by setting CUDA_VISIBLE_DEVICES env variable in
# order to prevent any memory allocation on unused GPUs

args.cuda = not args.no_cuda and torch.cuda.is_available()
np.random.seed(args.seed)

if not os.path.exists(args.log_dir):
    os.makedirs(args.log_dir)

if args.cuda:
    cudnn.benchmark = False
torch.backends.cudnn.enabled = True

LOG_DIR = args.log_dir
os.system('chmod -R 777 %s'%LOG_DIR)

if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)


def main():
    
    if args.is_training == 0:
        from model_use import ResNet_MIT
        from load_data import MyDataset
        
        if args.fold=='fold1':
            checklis=[20,30,36]
        elif args.fold=='fold2':
            checklis=[27,40,80]
        elif args.fold=='fold3':
            checklis=[17,32,42]
        elif args.fold=='fold6':
            checklis=[15,16,39]
            
        for checkpoint_number in checklis:
            model=ResNet_MIT()
            model.cuda()
            model = torch.nn.DataParallel(model)
                
            print('\nparsed options:\n{}\n'.format(vars(args)))
            print(LOG_DIR)
            optimizer = create_optimizer(model, args.lr)
            args.resume=LOG_DIR+'/model/checkpoint_'+str(checkpoint_number)+'.pth'
            # optionally resume from a checkpoint
            if args.resume:
                print('=> loading checkpoint {}'.format(args.resume))
                checkpoint = torch.load(args.resume)
                model.load_state_dict(checkpoint['state_dict'])
                epoch=checkpoint['epoch']

            # test_lis=['test_class']
            # for item in test_lis:
                # test_file1='./dataset/'+args.fold+'/'+item+'.scp'
                # test_set = MyDataset(test_file1,batch_size=1,train_flag=False)            
                # test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)
                # acc=test_class(model, epoch,test_loader,test_file1,checkpoint_number,item)
                
            final_test='./dataset/test_final.scp'
            final_test_set = MyDataset(final_test,batch_size=1,train_flag=False)            
            final_test_loader = DataLoader(final_test_set, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)
            acc1=final_test_class(model, epoch,final_test_loader,final_test,checkpoint_number,final_test)
    else:
        from model_use import ResNet_MIT
        from load_data import MyDataset
        model=ResNet_MIT()
        model.cuda()
        model = torch.nn.DataParallel(model)

        print('\nparsed options:\n{}\n'.format(vars(args)))
        print(LOG_DIR)
        loss_mean = []
        optimizer = create_optimizer(model, args.lr)
        scheduler=optim.lr_scheduler.StepLR(optimizer,step_size=20,gamma=0.5)
            
        start = args.start_epoch
        end = start + args.epochs
        
        train_file1='./dataset/'+args.fold+'/train_class.scp'
        valid_file1='./dataset/'+args.fold+'/valid_class.scp'
        test_flie1='./dataset/'+args.fold+'/test_class.scp'
        
        valid_set = MyDataset(valid_file1,batch_size=1,train_flag=False)            
        valid_loader = DataLoader(valid_set, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)
        test_set = MyDataset(test_flie1,batch_size=1,train_flag=False)            
        test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)
        
        auclist=[]
        for epoch in range(start, end):
            train_set = MyDataset(train_file1,batch_size=32,train_flag=True)            
            train_loader = DataLoader(train_set, batch_size=32, shuffle=True, num_workers=4, pin_memory=True,drop_last=False)
            train_spec1(model, optimizer, epoch, loss_mean,train_loader)
            scheduler.step()
            if epoch<300:
                loss_val=Valid_class( model, epoch,valid_loader,'valid_class')
                loss_test=Valid_class( model, epoch,test_loader,'test_class')
                torch.save({'epoch': epoch + 1, 'state_dict': model.state_dict()},
                            LOG_DIR+'/model'+'/checkpoint-'+str(epoch)+'-'+str(loss_val)+'-'+str(loss_test)+'.pth')

def train_spec1(model, optimizer, epoch, loss_mean,train_loader):
    model.train()
    criterion = torch.nn.CrossEntropyLoss()
    loss_tmp=0.0
    itr=0
    epoch_itr=len(train_loader)
    for batch,data_and_label in enumerate(train_loader):
        with torch.autograd.set_detect_anomaly(True):
            utt,data,label=data_and_label
            data,label=data.cuda(),label.cuda() 
            label=label.flatten().long()
            out_cls=model(data)
            out_cls=out_cls.float()
            loss = criterion(out_cls, label)
            optimizer.zero_grad()
            loss.backward()
            loss_tmp += loss.item()

        utils.clip_grad_value_(model.parameters(), args.init_clip_max_norm)
        optimizer.step()
        itr += 1
        if itr % args.log_interval == 0 and itr!=0:
            loss_tmp /= args.log_interval
            fp = open(LOG_DIR+'/model'+'/loss.txt','a')
            fp.write('Train Epoch: {:3d} [{:8d}/{:8d} ({:3.0f}%)]\tLoss: {:.6f}\tlr:{:.4f}\n'.format(
                epoch, itr, epoch_itr,
                100. * itr / epoch_itr,loss_tmp,
                 optimizer.param_groups[0]['lr']))
            fp.close()
            loss_mean.append(loss_tmp)
            loss_tmp = 0.0
    # do checkpointing
    if epoch<300:
        torch.save({'epoch': epoch + 1, 'state_dict': model.state_dict()},
                   '{}/checkpoint_{}.pth'.format(LOG_DIR+'/model', epoch))
        np.save(LOG_DIR+'/model'+'/loss_%s'%epoch, np.array(loss_mean))

def Valid_class( model, epoch,test_loader,setname):
    model.eval()    
    logfile=open(LOG_DIR+'/model/'+setname.split('_')[0]+'log.txt','w')
    criterion = nn.CrossEntropyLoss()
    total_num=0
    right_num=0
    agefile=open('./dataset/'+args.fold+'/'+setname.split('_')[0]+'.scp','r')
    lines=agefile.readlines()
    refdic={}
    mae=0
    for line in lines:
        EC,EO,age=line[:-1].split(' ')
        speaker=EC.split('subj')[1][:4]
        refdic[speaker]=float(age)

    for batch,data_and_label in enumerate(test_loader):
        utt,spec,label=data_and_label
        spec,label=spec.cuda(),label.cuda() 
        lab=label      
        total_num+=1
        with torch.no_grad():
            data=spec
            tmp= model(data)
        ind=torch.argmax(tmp)
        if ind == lab:
            right_num+=1
        indlis=find_nmax_index(tmp,2)
        indlis=np.array(indlis)
        mae+=abs(refdic[utt[0]]-(np.mean(indlis)+4))
        for i in range(len(utt)):
            logfile.write(str(utt[i])+' '+str(tmp[i])+'\n')
    logfile.close()   
    print('ACC: ',right_num/total_num)
    print('MAE: ',mae/len(lines))
    return mae/len(lines)

def test_class( model, epoch,test_loader,test_file,checkpoint_number,setname):
    model.eval()    
    filename=test_file.split('/')[-1].split('.')[0]
    logfile=open(LOG_DIR+filename+'_'+str(checkpoint_number)+'.txt','w')
    total_num=0
    right_num=0
    agefile=open('./dataset/'+args.fold+'/'+setname.split('_')[0]+'.scp','r')
    lines=agefile.readlines()
    refdic={}
    mae=0
    for line in lines:
        EC,EO,age=line[:-1].split(' ')
        speaker=EC.split('subj')[1][:4]
        refdic[speaker]=float(age)
    # print(refdic)
    for batch,data_and_label in enumerate(test_loader):
        utt,spec,label=data_and_label
        # print(utt)
        spec,label=spec.cuda(),label.cuda() 
        lab=label      
        total_num+=1
        with torch.no_grad():
            data=spec
            tmp= model(data)
        ind=torch.argmax(tmp)
        if ind == lab:
            right_num+=1
        indlis=find_nmax_index(tmp,2)
        indlis=np.array(indlis)
        mae+=abs(refdic[utt[0]]-(np.mean(indlis)+4))
        for i in range(len(utt)):
            logfile.write(str(utt[i])+' '+str(tmp[i])+'\n')
    logfile.close()   
    print('ACC: ',right_num/total_num)
    print('MAE: ',mae/len(lines))
    return mae/len(lines)
    
def find_nmax_index(a,n):
    a=a.view(-1)
    indlis=[]
    for i in range(n):
        indlis.append(int(torch.argmax(a)))
        a[indlis[-1]]=-100
    return indlis
    
def final_test_class( model, epoch,test_loader,test_file,checkpoint_number,setname):
    model.eval()    
    predict_lis=[]
    for batch,data_and_label in enumerate(test_loader):
        utt,spec,label=data_and_label
        spec,label=spec.cuda(),label.cuda() 
        lab=label      
        with torch.no_grad():
            data=spec
            tmp= model(data)
        indlis=find_nmax_index(tmp,2)
        indlis=np.array(indlis)
        predict_age=np.mean(indlis)+4
        predict_lis.append(predict_age)
    predict_lis=np.array(predict_lis)
    dummy_submission = []
    for subj, pred in zip(range(1601, 1601 + 400), predict_lis):
        dummy_submission.append({"id": subj, "age": pred})
    pd.DataFrame(dummy_submission).to_csv('./test_final_results/melspec_class_equalprob_average_'+args.fold+'_'+str(checkpoint_number)+'.csv', index=False)
    return 0
    
def create_optimizer(model, new_lr):
    # setup optimizer
    if args.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=new_lr,
                              momentum=0.9, dampening=0.9,
                              weight_decay=args.wd)
    elif args.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=new_lr,
                               weight_decay=args.wd)
    elif args.optimizer == 'adagrad':
        optimizer = optim.Adagrad(model.parameters(),
                                  lr=new_lr,
                                  lr_decay=args.lr_decay,
                                  weight_decay=args.wd)
    #optimizer = nn.DataParallel(optimizer, device_ids=device_ids)
    #torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=40, verbose=True, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08)
    
    return optimizer


if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')
    main()
