import math
from math import *
import numpy as np
import torchvision.models as models
import torch.utils.data as data
from torchvision import transforms
import cv2
import torch.nn.functional as F
from torch.autograd import Variable
import pandas as pd
import os ,torch
import torch.nn as nn
import image_utils
import argparse,random
import tqdm
from PIL import Image
#
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import itertools
#
from sklearn.metrics.pairwise import cosine_similarity


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='./datasets/raf-basic/', help='Raf-DB dataset path.')
    parser.add_argument('--train', action='store_true', help='run training')
    parser.add_argument('--test', action='store_true', help='run testing')
    parser.add_argument('--image_retrieval', action='store_true', help='run image retrieval')
    parser.add_argument('--query_path', type=str, default='', help='query image path.')
    parser.add_argument('--checkpoint', type=str, default='./models/epoch27_acc0.8735.pth',help='Pytorch checkpoint file path')
    parser.add_argument('--batch_size', type=int, default=60, help='Batch size.')
    parser.add_argument('--optimizer', type=str, default="adam", help='Optimizer, adam or sgd.')
    parser.add_argument('--lr', type=float, default=0.01, help='Initial learning rate for sgd.')
    parser.add_argument('--momentum', default=0.9, type=float, help='Momentum for sgd')
    parser.add_argument('--workers', default=4, type=int, help='Number of data loading workers (default: 4)')
    parser.add_argument('--epochs', type=int, default=100, help='Total training epochs.')
    parser.add_argument('--drop_rate', type=float, default=0, help='Drop out rate.')
    return parser.parse_args()
    
class RafDataSet(data.Dataset):
    def __init__(self, raf_path, phase, transform = None,flip_transform=None, basic_aug = False):
        self.phase = phase
        self.transform = transform
        self.flip_transform = flip_transform
        self.raf_path = raf_path

        NAME_COLUMN = 0
        LABEL_COLUMN = 1
        df = pd.read_csv(os.path.join(self.raf_path, 'EmoLabel/list_patition_label.txt'), sep=' ', header=None)
        if phase == 'train':
            dataset = df[df[NAME_COLUMN].str.startswith('train')]
        else:
            dataset = df[df[NAME_COLUMN].str.startswith('test')]
        file_names = dataset.iloc[:, NAME_COLUMN].values
        self.label = dataset.iloc[:, LABEL_COLUMN].values - 1 
        # 0:Surprise, 1:Fear, 2:Disgust, 3:Happiness, 4:Sadness, 5:Anger, 6:Neutral
        
        self.file_paths = []
        # use raf aligned images for training/testing
        for f in file_names:
            f = f.split(".")[0]
            f = f +"_aligned.jpg"
            path = os.path.join(self.raf_path, 'Image/aligned', f)
            self.file_paths.append(path)
        
        self.basic_aug = basic_aug
        self.aug_func = [image_utils.add_gaussian_noise]

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        path = self.file_paths[idx]
        image = cv2.imread(path)
        image = image[:, :, ::-1] # BGR to RGB
        label = self.label[idx]
        # augmentation
        if self.phase == 'train':
            if self.basic_aug and random.uniform(0, 1) > 0.5:
                index = random.randint(0,1)
                image = self.aug_func[index](image)
            
        if self.transform is not None:
            image_o = self.transform(image)
        if self.flip_transform is not None:
                image_f = self.flip_transform(image)
        
        return image_o, image_f, label
      
class chiral_model_v3(nn.Module):
    def __init__(self, backbone, num_classes, pretrained):
        super(chiral_model_v3, self).__init__()
        self.backbone = backbone
        if backbone=='resnet18':
            self.main = models.resnet18(pretrained)
            self.main.fc = nn.Linear(512,256)
            
        self.classifier1 = nn.Linear(128,2)
        self.classifier2 = nn.Linear(128,num_classes)
        
        self.avg = nn.AdaptiveAvgPool2d((1,1))
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x, flip_x):
        
        x = self.main(x)
        x_face = x.narrow(1,0,128)
        x_emo = x.narrow(1,128,128)
        x_face = self.classifier1(x_face)
        x_emo = self.classifier2(x_emo)
        
        flip_x = self.main(flip_x)
        flip_face = flip_x.narrow(1,0,128)
        flip_emo = flip_x.narrow(1,128,128)
        flip_face = self.classifier1(flip_face)
        flip_emo = self.classifier2(flip_emo)
        
        return x_face,flip_face,x_emo,flip_emo
  
# Custom Contrastive Loss
class ContrastiveLoss(torch.nn.Module):

    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2)
        loss_contrastive = torch.mean((1-label) * torch.pow(euclidean_distance, 2) +     # calmp夹断用法
                                      (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))     
 

        return loss_contrastive
 
def run_training():
    # Arguments
    args = parse_args()    

    # Build Model 
    model = chiral_model_v3('resnet18', 7, True)
    model = model.cuda()        
    
    # Prepare Training Data
    data_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(scale=(0.02,0.25))])
    data_transforms_flip = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(scale=(0.02,0.25))])
    train_dataset = RafDataSet(args.dataset, phase = 'train', transform = data_transforms,flip_transform = data_transforms_flip, basic_aug = False)
    print('Training set size:', train_dataset.__len__())
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size = args.batch_size,
                                               num_workers = args.workers,
                                               shuffle = True,  
                                               pin_memory = True)
    # Prepare Testing Data
    data_transforms_val = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])])   
    data_transforms_val_flip = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(p=1),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])])                                           
    val_dataset = RafDataSet(args.dataset, phase = 'test', transform = data_transforms_val, flip_transform=data_transforms_val_flip)    
    print('Validation set size:', val_dataset.__len__())
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                               batch_size = args.batch_size,
                                               num_workers = args.workers,
                                               shuffle = False,  
                                               pin_memory = True)
    

    # Define Loss Functions
    CE = torch.nn.CrossEntropyLoss()
    MSE = torch.nn.MSELoss()
    BCE = torch.nn.BCEWithLogitsLoss()
    CON = ContrastiveLoss()

    # Define Optimizer
    params = model.parameters()
    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(params,weight_decay = 1e-4)
    elif args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(params, args.lr,
                                    momentum=args.momentum,
                                    weight_decay = 1e-4)
    else:
        raise ValueError("Optimizer not supported.")
    
    # Adjusting Learning Rate
    #scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma = 0.9)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 20, gamma=0.1)
    

    best_val_acc = 0.0
    for i in range(1, args.epochs + 1):
        # training
        running_loss = 0.0
        running_emo = 0.0
        running_mse = 0.0
        running_constrative = 0.0
        running_bce = 0.0

        correct_sum = 0
        iter_cnt = 0
        model.train()
        
        for batch_i, (imgs, imgs_f, targets) in enumerate(train_loader):
            batch_sz = imgs.size(0) 
            iter_cnt += 1
            optimizer.zero_grad()

            imgs = imgs.cuda()
            imgs_f = imgs_f.cuda()
            targets = targets.cuda()
            face_t = torch.tensor([1.,0.]).unsqueeze(0).repeat((imgs.shape[0],1)).cuda()
            flip_t = torch.tensor([0.,1.]).unsqueeze(0).repeat((imgs.shape[0],1)).cuda()
            
            face, flip_face, emo, flip_emo = model(imgs,imgs_f)
            
            emo_loss = CE(emo, targets) + CE(flip_emo,targets)
            face_loss = BCE(face,face_t) + BCE(flip_face,flip_t)
            mse_loss =  MSE(emo,flip_emo)
            constrative_loss = CON(face, flip_face,1)
            loss = emo_loss + mse_loss + constrative_loss + face_loss
            
            loss.backward()
            optimizer.step()

            running_loss += loss
            running_emo += emo_loss
            running_mse += mse_loss
            running_constrative += constrative_loss
            running_bce += face_loss
            _, predicts = torch.max((emo+flip_emo), 1)
            correct_num = torch.eq(predicts, targets).sum()
            correct_sum += correct_num

        scheduler.step()
        acc = correct_sum.float() / float(train_dataset.__len__())
        running_loss = running_loss/iter_cnt
        print('[Epoch %d] Training accuracy: %.4f. Loss: %.3f '\
                     % (i, acc, running_loss))
        
        # testing
        with torch.no_grad():
            
            running_loss = 0.0
            iter_cnt = 0
            bingo_cnt = 0
            sample_cnt = 0
            model.eval()
            for batch_i, (imgs, imgs_f, targets) in enumerate(val_loader):
                imgs = imgs.cuda()
                imgs_f = imgs_f.cuda()
                targets = targets.cuda()

                face, flip_face, emo, flip_emo = model(imgs,imgs_f)

                loss = CE(emo, targets) + CE(flip_emo,targets) + MSE(emo,flip_emo) + CON(face, flip_face,1)
                
                running_loss += loss
                iter_cnt+=1
                _, predicts = torch.max((emo+flip_emo), 1)
                correct_num  = torch.eq(predicts,targets)
                bingo_cnt += correct_num.sum().cpu()
                sample_cnt += emo.size(0)
                
            running_loss = running_loss/iter_cnt   
            acc = bingo_cnt.float()/float(sample_cnt)
            acc = np.around(acc.numpy(),4)
            print("[Epoch %d] Validation accuracy:%.4f. Loss:%.3f" % (i, acc, running_loss))
            
            if acc > best_val_acc:
                best_val_acc = acc
                print("############best_val_acc##############",best_val_acc)
                if acc > 0.86:
                    torch.save({'iter': i,
                                'model_state_dict': model.state_dict(),
                                'optimizer_state_dict': optimizer.state_dict(),},
                                os.path.join('models', "epoch"+str(i)+"_acc"+str(acc)+".pth"))
                    print('Model saved.')

    
def testing(ckpt_path):
    model = chiral_model_v3('resnet18', 7, True)
    checkpoint = torch.load(ckpt_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.cuda()
    model.eval()
    data_transforms_val = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])])   
    data_transforms_val_flip = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(p=1),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])])                                           
    val_dataset = RafDataSet(args.dataset, phase = 'test', transform = data_transforms_val, flip_transform=data_transforms_val_flip)    
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                               batch_size = args.batch_size,
                                               num_workers = args.workers,
                                               shuffle = False,  
                                               pin_memory = True)

    y_pred = []
    y_true = []
    with torch.no_grad():
        running_loss = 0.0
        iter_cnt = 0
        bingo_cnt = 0
        sample_cnt = 0
        model.eval()
        for batch_i, (imgs, imgs_f, targets) in enumerate(val_loader):
            imgs = imgs.cuda()
            imgs_f = imgs_f.cuda()
            targets = targets.cuda()

            face, flip_face, emo, flip_emo = model(imgs,imgs_f)

            iter_cnt+=1
            _, predicts = torch.max((emo+flip_emo), 1)

            y_pred+=predicts.cpu()
            y_true+=targets.cpu()
            
            correct_num  = torch.eq(predicts,targets)
            bingo_cnt += correct_num.sum().cpu()
            sample_cnt += emo.size(0)
            
        acc = bingo_cnt.float()/float(sample_cnt)
        acc = np.around(acc.numpy(),4)
        print("Validation accuracy:%.4f." %acc)
        

class RafDataSet_r(data.Dataset):
    def __init__(self, raf_path, phase, transform = None,flip_transform=None, basic_aug = False):
        self.phase = phase
        self.transform = transform
        self.flip_transform = flip_transform
        self.raf_path = raf_path

        NAME_COLUMN = 0
        LABEL_COLUMN = 1
        df = pd.read_csv(os.path.join(self.raf_path, 'EmoLabel/list_patition_label.txt'), sep=' ', header=None)
        if phase == 'train':
            dataset = df[df[NAME_COLUMN].str.startswith('train')]
        else:
            dataset = df[df[NAME_COLUMN].str.startswith('test')]
        file_names = dataset.iloc[:, NAME_COLUMN].values
        self.label = dataset.iloc[:, LABEL_COLUMN].values - 1 # 0:Surprise, 1:Fear, 2:Disgust, 3:Happiness, 4:Sadness, 5:Anger, 6:Neutral
        
        self.file_paths = []
        # use raf aligned images for training/testing
        for f in file_names:
            f = f.split(".")[0]
            f = f +"_aligned.jpg"
            path = os.path.join(self.raf_path, 'Image/aligned', f)
            self.file_paths.append(path)
        
        self.basic_aug = basic_aug
        self.aug_func = [image_utils.add_gaussian_noise]

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        path = self.file_paths[idx]
        image = cv2.imread(path)
        image = image[:, :, ::-1] # BGR to RGB
        label = self.label[idx]
                   

        if self.transform is not None:
            image_o = self.transform(image)
        if self.flip_transform is not None:
                image_f = self.flip_transform(image)
        
        return image_o, image_f, label, path

class chiral_model_v3_r(nn.Module):
    def __init__(self, backbone, num_classes, pretrained):
        super(chiral_model_v3_r, self).__init__()
        self.backbone = backbone
        if backbone=='resnet18':
            self.main = models.resnet18(pretrained)
            self.main.fc = nn.Linear(512,256)
            
        self.classifier1 = nn.Linear(128,2)
        self.classifier2 = nn.Linear(128,7)
        
        self.avg = nn.AdaptiveAvgPool2d((1,1))
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x, flip_x):
        
        x = self.main(x)
        x_face = x.narrow(1,0,128)
        x_emo = x.narrow(1,128,128)
        x_face = self.classifier1(x_face)
        x_emo = self.classifier2(x_emo)
        
        flip_x = self.main(flip_x)
        flip_face = flip_x.narrow(1,0,128)
        flip_emo = flip_x.narrow(1,128,128)
        flip_face = self.classifier1(flip_face)
        flip_emo = self.classifier2(flip_emo)
        


        return x,flip_x

def img_retrieval(query_path):
    args = parse_args()
    
    ckpt_path = args.checkpoint
    
    feature = []
    def get_activation():
        def hook(model, input, output):
            feature.append(output.detach())
        return hook

    model = chiral_model_v3_r('resnet18', 7, True)
    checkpoint = torch.load(ckpt_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    model.eval()

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])])
    transforms_flip = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(p=1),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])])

    # get query feature
    image = cv2.imread(query_path)
    image = image[:, :, ::-1] # BGR to RGB    
    q = transform(image).unsqueeze(0)
    q_f = transforms_flip(image).unsqueeze(0)
   
    x,x_f = model (q,q_f)
    query = x.detach()
    query_f = x_f.detach()
    
    #
    val_dataset = RafDataSet_r(args.dataset, phase = 'test', transform = transform, flip_transform=transforms_flip)   
    train_dataset = RafDataSet_r(args.dataset, phase = 'train', transform = transform, flip_transform=transforms_flip)   
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size = 64,
                                               num_workers = args.workers,
                                               shuffle = False,  
                                               pin_memory = True)
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                               batch_size = 64,
                                               num_workers = args.workers,
                                               shuffle = False,  
                                               pin_memory = True)
    
    paths = []
    features = []
    features_f = []
    model.cuda()
    for batch_i, (imgs, imgs_f, targets,path) in enumerate(val_loader):
        imgs = imgs.cuda()
        imgs_f = imgs_f.cuda()

        feature, feature_f = model(imgs,imgs_f)
        paths+=(path)
        features+=(feature.detach().cpu())
        features_f+=(feature_f.detach().cpu())
    for batch_i, (imgs, imgs_f, targets,path) in enumerate(train_loader):
        imgs = imgs.cuda()
        imgs_f = imgs_f.cuda()

        feature, feature_f = model(imgs,imgs_f)
        paths+=(path)
        features+=(feature.detach().cpu())
        features_f+=(feature_f.detach().cpu())
        

    dis = []
    for i,f in enumerate(features):
        a = cosine_similarity(query.numpy(), [f.numpy()])
        b = cosine_similarity(query.numpy(), [features_f[i].numpy()])
        dis.append(a+b)

    best = sorted(range(len(dis)), key=lambda i: dis[i], reverse=True)[:11]
    for i in best:
        img_path = paths[i]
        save_dir = './image_retrieval/'
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        im =  Image.open(img_path)
        path = os.path.join(save_dir,img_path.split('/')[-1])
        im.save(path)
    im =  Image.open(query_path)
    path = os.path.join(save_dir,'query.jpg')
    im.save(path)


if __name__ == "__main__":                    
    args = parse_args()
    ckpt_path = args.checkpoint
    if args.train == True:
        # training
        run_training()
    elif args.test==True:
        # Test on test set
        # Prepare Testing Data
        testing(ckpt_path)
        
    elif args.image_retrieval==True:
        query_path = args.query_path
        if query_path == '':
            print("please insert query image path")
            assert 0
        img_retrieval(query_path)
    else:
        print("choose the desired action from train, test and image_retrieval")
        assert 0
   
    
    
    

