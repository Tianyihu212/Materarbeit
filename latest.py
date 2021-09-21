# -*- coding: utf-8 -*-
"""latest.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1HEV85uP2p47h5N6i31qx53LcWliWzc4b
"""

!pip install torch
!pip install torchvision
!pip install numpy
!pip install pandas
!pip install pillow
!pip install boto3
!pip install opencv-python
!pip install fsspec
!pip install s3fs

import os
import io

import cv2
import boto3
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
import torchvision.models as models

os.environ['AWS_ACCESS_KEY_ID'] = 'AKIA4I7AITZ4Q4WXYN5V'
os.environ['AWS_SECRET_ACCESS_KEY'] = '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'

s3 = boto3.resource('s3')

def read_image_from_s3(key):
    bucket = s3.Bucket('masterarbeit125255aa')
    img = bucket.Object(key).get().get('Body').read()
    nparray = cv2.imdecode(np.asarray(bytearray(img)), cv2.IMREAD_COLOR)
    return Image.fromarray(nparray)

def read_csv_from_s3(stage):
    if stage=='train':
        df = pd.read_csv('s3://masterarbeit125255aa/data/train/train.csv')
    elif stage=='valid':
        df = pd.read_csv('s3://masterarbeit125255aa/data/train/valid.csv')
    else:
        raise ValueError(f'not supported stage{stage}')
    return df

class SiameseGLDV2(Dataset):
    """
    Train: For each sample creates a positive or a negative pair
    Test: For each sample creates a positive or a negative pair
    """
    def __init__(self, stage: str, inferance=False):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.df = read_csv_from_s3(stage)
        self.df.drop(self.df.filter(regex="Unname"),axis=1, inplace=True)
        print(f'shape of df is {len(self.df)}, stage is {stage}')
        self.s3 = boto3.resource('s3')
        self.s3path = 'data/train/train_compress'
        if not inferance:
            self.my_transformer = transforms.Compose([
              transforms.RandomPerspective(distortion_scale=0.5, p=0.5, fill=0),
              transforms.ColorJitter(brightness=0, contrast=0, saturation=0, hue=0),
              transforms.RandomRotation(degrees=(0, 180), expand=False, center=None, fill=0, resample=None),
              transforms.RandomCrop(size=(224, 224), padding=None, pad_if_needed=False, fill=0, padding_mode='constant'),
              transforms.ToTensor(),
              transforms.RandomErasing(),
              transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])
        else:
            self.my_transformer = transforms.Compose([
              transforms.ToTensor(),
              transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])


    def __len__(self):
        return len(self.df)
    

    def plot(self, index: int):
        from IPython.display import Image as Image2
        from IPython.display import display
        # get an image from CSV file
        label = self.df.iloc[index]['landmark_id']
        anchor = self.df.iloc[index]['anchor'].split('\\')
        anchor_class = anchor[1]
        anchor_filen = anchor[2]
        pair = self.df.iloc[index]['uri'].split('\\')
        pair_class = pair[1]
        pair_filen = pair[2]
        anchor_image = self.s3path + '/' + anchor_class + '/' + anchor_filen
        pair_image = self.s3path + '/' + pair_class + '/' + pair_filen
        anchor_im = read_image_from_s3(anchor_image)
        pair_im = read_image_from_s3(pair_image)
        target = self.df.iloc[index]['target']
        print(f'Label is {target}')
        display(anchor_im, pair_im)


    def __getitem__(self, index):
        # get an image from CSV file
        label = self.df.iloc[index]['landmark_id']
        anchor = self.df.iloc[index]['anchor'].split('\\')
        anchor_class = anchor[1]
        anchor_filen = anchor[2]
        pair = self.df.iloc[index]['uri'].split('\\')
        pair_class = pair[1]
        pair_filen = pair[2]
        anchor_image = self.s3path + '/' + anchor_class + '/' + anchor_filen
        pair_image = self.s3path + '/' + pair_class + '/' + pair_filen
        anchor_im = read_image_from_s3(anchor_image)
        pair_im = read_image_from_s3(pair_image)
        transformed_anchor_im = self.my_transformer(anchor_im)
        transformed_pair_im = self.my_transformer(pair_im)
        target = self.df.iloc[index]['target']
        return (transformed_anchor_im, transformed_pair_im), target # target 1 means it's positive, otherwise netgative

train_dataset = SiameseGLDV2(stage='train')
valid_dataset = SiameseGLDV2(stage='valid')
batch_size = 128
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

class EmbeddingNet(nn.Module):
    def __init__(self, conv_net):
        super().__init__()
        self.convnet = conv_net
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2048, 1024),
            nn.PReLU(),
            nn.Linear(1024, 512),
            nn.PReLU(),
            nn.Linear(512, 256),
            nn.PReLU(),
            nn.Linear(256, 2)
        )

    def forward(self, x):
        output = self.convnet(x)
        output = output.view(output.size()[0], -1)
        output = self.fc(output)
        return output

    def get_embedding(self, x):
        return self.forward(x)

class SiameseNet(nn.Module):
    def __init__(self, embedding_net):
        super().__init__()
        self.embedding_net = embedding_net

    def forward(self, x1, x2):
        output1 = self.embedding_net(x1)
        output2 = self.embedding_net(x2)
        return output1, output2

    def get_embedding(self, x):
        return self.embedding_net(x)


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss
    Takes embeddings of two samples and a target label == 1 if samples are from the same class and label == 0 otherwise
    """

    def __init__(self, margin):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.eps = 1e-9

    def forward(self, output1, output2, target, size_average=True):
        # distances = (output2 - output1).pow(2).sum(1)  # squared distances
        distances = torch.cdist(output1, output2, p=2)
        losses = 0.5 * (target.float() * distances +
                        (1 + -1 * target).float() * F.relu(self.margin - (distances + self.eps).sqrt()).pow(2))
        return losses.mean()

cuda = torch.cuda.is_available()
conv_net = models.resnet50(pretrained=False, progress=True)
conv_net  = torch.nn.Sequential(*(list(conv_net.children())[:-1]))
embedding_net = EmbeddingNet(conv_net=conv_net)
# Step 3
model = SiameseNet(embedding_net)
if cuda:
    model.cuda()

margin = 1.
loss_fn = ContrastiveLoss(margin)
lr = 0.1
optimizer = optim.Adam(model.parameters(), lr=lr)
scheduler = lr_scheduler.StepLR(optimizer, 8, gamma=0.1, last_epoch=-1)
n_epochs = 20
log_interval = 500

import torch
import numpy as np
train_losses= []
valid_losses= []

def fit(train_loader, val_loader, model, loss_fn, optimizer, scheduler, n_epochs, cuda, log_interval, metrics=[],
        start_epoch=0):
    """
    Loaders, model, loss function and metrics should work together for a given task,
    i.e. The model should be able to process data output of loaders,
    loss function should process target output of loaders and outputs from the model

    Examples: Classification: batch loader, classification model, NLL loss, accuracy metric
    Siamese network: Siamese loader, siamese model, contrastive loss
    Online triplet learning: batch loader, embedding model, online triplet loss
    """
    for epoch in range(0, start_epoch):
        scheduler.step()

    for epoch in range(start_epoch, n_epochs):
        scheduler.step()

        # Train stage
        train_loss, metrics = train_epoch(train_loader, model, loss_fn, optimizer, cuda, log_interval, metrics)

        message = 'Epoch: {}/{}. Train set: Average loss: {:.4f}'.format(epoch + 1, n_epochs, train_loss)
        for metric in metrics:
            message += '\t{}: {}'.format(metric.name(), metric.value())

        val_loss, metrics = test_epoch(val_loader, model, loss_fn, cuda, metrics)

        message += '\nEpoch: {}/{}. Validation set: Average loss: {:.4f}'.format(epoch + 1, n_epochs,
                                                                                 val_loss)
        for metric in metrics:
            message += '\t{}: {}'.format(metric.name(), metric.value())

        print(message)


def train_epoch(train_loader, model, loss_fn, optimizer, cuda, log_interval, metrics):
    for metric in metrics:
        metric.reset()

    model.train()
    losses = []
    total_loss = 0
    num_trained = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        num_trained += batch_size
        print(f"Train Epoch: current batch is {batch_idx} with {num_trained} images.")
        target = target if len(target) > 0 else None
        if not type(data) in (tuple, list):
            data = (data,)
        if cuda:
            data = tuple(d.cuda() for d in data)
            if target is not None:
                target = target.cuda()


        optimizer.zero_grad()
        outputs = model(*data)

        if type(outputs) not in (tuple, list):
            outputs = (outputs,)

        loss_inputs = outputs
        if target is not None:
            target = (target,)
            loss_inputs += target

        loss_outputs = loss_fn(*loss_inputs)
        loss = loss_outputs[0] if type(loss_outputs) in (tuple, list) else loss_outputs
        losses.append(loss.item())
        total_loss += loss.item()
        loss.backward()
        optimizer.step()

        for metric in metrics:
            metric(outputs, target, loss_outputs)

        if batch_idx % log_interval == 0:
            message = 'Train: [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                batch_idx * len(data[0]), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), np.mean(losses))
            for metric in metrics:
                message += '\t{}: {}'.format(metric.name(), metric.value())

            print(message)
            losses = []

    total_loss /= (batch_idx + 1)
    train_losses.append(total_loss)
    return total_loss, metrics


def test_epoch(val_loader, model, loss_fn, cuda, metrics):
    num_trained = 0
    with torch.no_grad():
        total_loss = 0
        for metric in metrics:
            metric.reset()
        model.eval()
        val_loss = 0
        for batch_idx, (data, target) in enumerate(val_loader):
            num_trained += batch_size
            print(f"Valid Epoch: current batch is {batch_idx} with {num_trained} images.")
            target = target if len(target) > 0 else None
            if not type(data) in (tuple, list):
                data = (data,)
            if cuda:
                data = tuple(d.cuda() for d in data)
                if target is not None:
                    target = target.cuda()

            outputs = model(*data)

            if type(outputs) not in (tuple, list):
                outputs = (outputs,)
            loss_inputs = outputs
            if target is not None:
                target = (target,)
                loss_inputs += target

            loss_outputs = loss_fn(*loss_inputs)
            loss = loss_outputs[0] if type(loss_outputs) in (tuple, list) else loss_outputs
            val_loss += loss.item()
            total_loss += loss.item()

            for metric in metrics:
                metric(outputs, target, loss_outputs)

    total_loss /= (batch_idx + 1)
    valid_losses.append(total_loss)
    return total_loss, metrics



import warnings
warnings.filterwarnings("ignore")
fit(train_loader, valid_loader, model, loss_fn, optimizer, scheduler, n_epochs, cuda, log_interval)

torch.save(model.state_dict(), 'model.pth')

import pandas as pd
df = pd.DataFrame({'train_loss':train_losses, 'valid_loss':valid_losses})
df.to_csv('loss.csv')

# train_dataset = SiameseGLDV2(stage='train', inferance=True)

# !pip install torch
# import torch
# print(torch.cuda.is_available())

# import torchvision.models as models
# import torch

# conv_net = models.resnet50(pretrained=False, progress=True)
# conv_net  = torch.nn.Sequential(*(list(conv_net.children())[:-1]))
# embedding_net = EmbeddingNet(conv_net=conv_net)
# # Step 3
# model = SiameseNet(embedding_net)

# model.load_state_dict(torch.load('/content/model.pth'))
# model.eval()


# with torch.no_grad():
#     (left, right), label = train_dataset[3]
#     tensor_2= right
#     output1,output2= model(left.unsqueeze(0), right.unsqueeze(0))
#     distances = torch.cdist(output1, output2, p=2)
#     # predicted_class = np.argmax(predit)
#     # print(predicted_class)
#     print(label)
#     print(distances)

# train_dataset.plot(0)

