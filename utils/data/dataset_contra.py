import numpy as np
import os
from PIL import Image
from PIL import ImageFile
from torchvision import transforms 
import pdb
from torch.utils.data import Dataset
import torch
import pickle as pkl
import cv2
import csv
import pandas as pd
from itertools import islice
import random

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

        
        self.train_dict = {}
        for i in range(len(self.train_list)):
            label_id = self.train_list[i][1]
            if label_id in self.train_dict.keys():
                self.train_dict[label_id].append(self.train_list[i])
            else:
                self.train_dict[label_id] = [self.train_list[i]]
                
        if self.subset == 'train':
            print('samples of train: ', len(self.train_list) )
        elif self.subset == 'val':
            print('samples of val: ', len(self.val_list) )
            
        elif self.subset == 'index':
            print('samples of index: ', len(self.index_list) )
        elif self.subset == 'test':
            print('samples of test: ', len(self.test_list) )
            
    def gen_train_sample_list(self):
        sample_list = []
        for i in range(len(self.train_list)):
            idx = self.train_list[i][1]
            sample_list.append(1/len(self.train_dict[idx]))
        return sample_list
    def __getitem__(self, index):
        if self.subset == 'train':
            img_path, label_id = self.train_list[index]
            img = Image.open(img_path)
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
            
            p = random.random()

            if p >0.5:
                flag = 1  # pos
                cnt = 0
                while True: 
                    cnt +=1
                    itm = random.sample(self.train_dict[label_id], 1)[0]
                    img_path2, label_id2 = itm
                    if img_path2 != img_path:
                        break
                    if cnt ==5:
                        break
            else:
                flag = 0  # neg
                label_list = list(set(range(7770)) - set([label_id]))
                # label_id2 = (label_id + 1) if label_id < (7770-1) else (label_id -1
                label_id2 = random.sample(label_list, 1)[0]
                itm = random.sample(self.train_dict[label_id2], 1)[0]
                img_path2, label_id2 = itm

            img2 = Image.open(img_path2)
            if img2.mode != 'RGB':
                img2 = img2.convert("RGB") 
            img2 = transforms.Resize((resize, resize), Image.BILINEAR)(img2)
            img2 = transforms.RandomCrop(self.input_size)(img2) 
            img2 = transforms.RandomHorizontalFlip()(img2)
            img2 = transforms.RandomRotation(degrees=15)(img2)
            # img2 = AutoAugImageNetPolicy()(img2)
            img2 = transforms.ToTensor()(img2)
            img2 = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(img2)

            
            sample = {'img': img, 'label': label_id,
                      'img2': img2, 'label2': label_id2, 'flag':flag  }

        elif self.subset == 'val':
            img_path, label = self.val_list[index]
            img = Image.open(img_path)

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
            img = transforms.ToTensor()(img)
            img = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(img)
            # img = Cutout(n_holes=4, length=20)(img)
            sample = {'img': img, 'label': label}
        elif self.subset == 'test':
            img_path, label = self.test_list[index]
            img = Image.open(img_path)
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