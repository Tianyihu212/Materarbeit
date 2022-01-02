import numpy as np
# import scipy.misc
# import imageio 
import os
from PIL import Image
from PIL import ImageFile
# ImageFile.LOAD_TRUNCATED_IMAGES = True
from torchvision import transforms 
import pdb
# import layers as aug
from torch.utils.data import Dataset
import torch
# from .auto_augment import AutoAugImageNetPolicy
# from .cutout import Cutout
import pickle as pkl
import cv2
import csv
import pandas as pd
from itertools import islice
class GLDDataset(Dataset):
    def __init__(self, root, input_size=224, subset=None, data_len=None):
        self.root = root#root
        self.input_size = input_size
        self.subset = subset
        # train_list_pkl = 'data/train_list.pkl'
        index_list_csv = 'index_final.csv'
        test_list_csv = 'test_final.csv'
        train_list_csv = 'split_train_final.csv'
        val_list_csv = 'split_val_final.csv'
        # label2class = 'label2class.csv'
        self.index_list = []
        self.test_list = []
        self.train_list = []
        self.val_list = []
        label2class = pd.read_csv('label2class_final.csv' ,header=None )
        label2class = dict(zip(label2class[0],label2class[1]))

        with open(train_list_csv)as f:
            f_csv = csv.reader(f)
            for row in f_csv:
#                 print(row)
                label = int(row[1])
                class_id = label2class[label]
                img_name = row[0]+'.jpg'
                img_path = os.path.join(self.root, img_name[0], img_name[1], img_name[2], img_name)
                self.train_list.append((img_path, class_id))
        with open(val_list_csv)as f:
            f_csv = csv.reader(f)
            for row in f_csv:
                label = int(row[1])
                class_id = label2class[label]
                img_name = row[0]+'.jpg'
                img_path = os.path.join(self.root, img_name[0], img_name[1], img_name[2], img_name)
                self.val_list.append((img_path, class_id))
        with open(index_list_csv)as f:
            f_csv = csv.reader(f)
            for row in islice(f_csv, 1, None):
                label = int(row[0])
                img_name = row[1].split('\\')[-1].split('.')[0]+'.jpg'
                img_path = os.path.join(self.root, img_name[0], img_name[1], img_name[2], img_name)
                self.index_list.append((img_path, label))
#         print(len(self.index_list))
        with open(test_list_csv)as f:
            f_csv = csv.reader(f)
            for row in islice(f_csv, 1, None):
                label = row[0]
                label = [int(x) for x in label.split()]
#                 print(label)
                img_name = row[1].split('\\')[-1].split('.')[0]+'.jpg'
#                 img_path = os.path.join(self.root, img_name[0], img_name[1], img_name[2], img_name)
                img_path = os.path.join(self.root, img_name)
                self.test_list.append((img_path, label))
        if self.subset == 'train':
            print('samples of train: ', len(self.train_list) )
        elif self.subset == 'val':
            print('samples of val: ', len(self.val_list) )
            
        elif self.subset == 'index':
            print('samples of index: ', len(self.index_list) )
        elif self.subset == 'test':
            print('samples of test: ', len(self.test_list) )
            
        
    def __getitem__(self, index):
        if self.subset == 'train':
            img_path, label = self.train_list[index]
            img = Image.open(img_path)
            # except:
            #     print(self.train_file_list[index])
                
            # if len(img.shape) == 2:
            #     img = np.stack([img] * 3, 2)
            if img.mode != 'RGB':
                img = img.convert("RGB") 
            if self.input_size == 224:
                resize = 256
            img = transforms.Resize((resize, resize), Image.BILINEAR)(img)
            img = transforms.RandomCrop(self.input_size)(img)   
            img = transforms.RandomHorizontalFlip()(img)
            img = transforms.RandomRotation(degrees=15)(img)
#             img = AutoAugImageNetPolicy()(img)
            img = transforms.ToTensor()(img)
            img = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(img)
            sample = {'img': img, 'label': label}

        elif self.subset == 'val':
            img_path, label = self.val_list[index]
            img = Image.open(img_path)
            # try:
            #     img = Image.open(img_path)
            # except:
            #     print(self.test_file_list[index])
            if img.mode != 'RGB':
                img = img.convert("RGB") 
        
            img = transforms.Resize((self.input_size, self.input_size), Image.BILINEAR)(img)
            # img = transforms.CenterCrop(self.input_size)(img)
            img = transforms.ToTensor()(img)
            img = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(img)
            sample = {'img': img, 'label': label}
        elif self.subset == 'index':
            # self.train_img = [Image.open(train_file) for train_file in self.train_file_list]
            # self.train_label = [x for i, x in zip(train_test_list, label_list) if i]
            # self.train_imgname = [x for x in train_file_list]
            # self.train_file_list[index]
            img_path, label = self.index_list[index]
            img = Image.open(img_path)
            # except:
            #     print(self.train_file_list[index])
                
            # if len(img.shape) == 2:
            #     img = np.stack([img] * 3, 2)
            if img.mode != 'RGB':
                img = img.convert("RGB") 
            img = transforms.Resize((self.input_size, self.input_size), Image.BILINEAR)(img)
            # img = transforms.RandomCrop(self.input_size)(img)
            # img = np.array(img)
            # img= cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
            # img = cv2.resize(img, (256, 256), interpolation = cv2.INTER_LINEAR) #INTER_LINEAR
            # img= cv2.cvtColor(np.asarray(img), cv2.COLOR_BGR2RGB)
            # img = Image.fromarray(img, mode='RGB')
            # img = img.convert("RGB")    
            # img = transforms.RandomHorizontalFlip()(img)
            # img = transforms.RandomRotation(degrees=15)(img)
            # img = AutoAugImageNetPolicy()(img)
            img = transforms.ToTensor()(img)
            img = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(img)
            # img = Cutout(n_holes=4, length=20)(img)
            sample = {'img': img, 'label': label}
        elif self.subset == 'test':
            img_path, label = self.test_list[index]
            img = Image.open(img_path)
            # try:
            #     img = Image.open(img_path)
            # except:
            #     print(self.test_file_list[index])
            if img.mode != 'RGB':
                img = img.convert("RGB") 
        
            img = transforms.Resize((self.input_size, self.input_size), Image.BILINEAR)(img)
            # img = transforms.CenterCrop(self.input_size)(img)
            img = transforms.ToTensor()(img)
            img = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(img)

            sample = {'img': img }
        return sample

    def __len__(self):
        if self.subset== 'train':
            return len(self.train_list)
#             return 256
        elif self.subset=='val':
            return len(self.val_list)
#             return 256
        elif self.subset== 'index':
            return len(self.index_list)
        elif self.subset=='test':
            return len(self.test_list)