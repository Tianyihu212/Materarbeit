# -*- coding: utf-8 -*-
"""
Create this train module by Tianyi Hu.
This project can be achieved through colab
"""

# here is environment
# !pip install torch
# !pip install torchvision
# !pip install numpy
# !pip install pandas
# !pip install pillow
# !pip install boto3
# !pip install opencv-python
# !pip install fsspec
# !pip install s3fs

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
torch.cuda.empty_cache()

os.environ['AWS_ACCESS_KEY_ID'] = 'Here is the s3 network disk account where I saved the data set'
os.environ['AWS_SECRET_ACCESS_KEY'] = 'Here is the s3 network disk password for saving the data set'
s3 = boto3.resource('s3')

def read_image_from_s3(key):
    """

    Parameters
    ----------
    key : string
          a image from dataset

    Returns
    -------
    Image.fromarray : np.array
                    load image from dataset
    """
    bucket = s3.Bucket('Here is s3 save address')
    img = bucket.Object(key).get().get('Body').read()
    nparray = cv2.imdecode(np.asarray(bytearray(img)), cv2.IMREAD_COLOR)
    return Image.fromarray(nparray)

def read_csv_from_s3(stage):
    """

    Parameters
    ----------
    stage : string
    distinguish train valid final csv.

    Returns
    -------
    df : pandas
        load csv file

    """
    if stage=='train':
        df = pd.read_csv('train.csv save address')
    elif stage=='valid':
        df = pd.read_csv('valid.csv save address')
    elif stage == 'final':
        df = pd.read_csv('final.csv save address')
    else:
        raise ValueError(f'not supported stage{stage}')
    return df

class SiameseGLDV2(Dataset):
    """
    This is the class of the GLD-v2 data set, which is used as training data for neutral network.
    The GLD-v2 data set contains over 1 million images of different landmark.
    uri of GLD-v2 data set: https://github.com/cvdfoundation/google-landmark

    Attributes :
        df : pandas framework, loaded csv file
        s3 : loaded AWS s3 web
        s3path : load AWS s3 address
        my_transformer : torchvision.transformers, mask for data augmentation
    """
    def __init__(self, stage: str, inferance=False):
        """

        Parameters
        ----------
        stage : string
                which csv file
        inferance : bool
                    if need data augmentation

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
        """
        Output the length of the data set read

        Returns
        -------
        self.df : int
                  dataset length
        """
        return len(self.df)
    

    def plot(self, index: int):
        """
        plot image pairs for
        Parameters
        ----------
        index

        Returns
        -------

        """
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
        """
        get a pair of images from CSV file

        Parameters
        ----------
        index : Which image in the dataset

        Returns
        -------
        transformed_anchor_im : query image
        transformed_pair_im : matching image
        """

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
final_dataset = SiameseGLDV2(stage='final')
batch_size = 16
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

cuda=False
for batch_idx, (data, target) in enumerate(train_loader):
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
        break

class EmbeddingNet(nn.Module):
    """
    A network EmbeddingNet for extracting features is defined.

    Attributes:
        conv_net : Convolutional layer of the network
        fc ：Fully connected layer of the network
    """
    def __init__(self, conv_net):
        """
        Initialize the EmbeddingNet

        Parameters
        ----------
        conv_net : nn.Module
                   Input in initialize network
        """
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
        """
        Sequence of Neural Network Operation

        Parameters
        ----------
        x : tensor
            Image to be embedded

        Returns
        -------
        output : tensor
                2d vector, output of neural network
        """
        output = self.convnet(x)
        output = output.view(output.size()[0], -1)
        output = self.fc(output)
        return output

    def get_embedding(self, x):
        """
        Call forward function

        Parameters
        ----------
        x : tensor
            Image to be embedded

        Returns
        -------
        forward(x) : tensor
                2d vector, output of neural network
        """
        return self.forward(x)

class SiameseNet(nn.Module):
    """
    SiameseNet is the share weight from a pair of image
    Output features of two images( a pair )

    Attributes:
        embedding_net : nn.Module
        extract feature of a image

    """
    def __init__(self, embedding_net):
        """
        Initialize the SiameseNet

        Parameters
        ----------
        embedding_net ：nn.Module
                       Extract feature neural network
        """
        super().__init__()
        self.embedding_net = embedding_net

    def forward(self, x1, x2):
        """
        Sequence of Neural Network Operation

        Parameters
        ----------
        x1 : tensor
         a image
        x2 : tensor
         a image

        Returns
        -------
        output1 : tensor
                feature of image
        output2 : tensor
                feature of image

        """
        output1 = self.embedding_net(x1)
        output2 = self.embedding_net(x2)
        return output1, output2

    def get_embedding(self, x):
        """
        Output feature of an image
        Parameters
        ----------
        x : tensor
            a image

        Returns
        -------
        embedding_net(x) : tensor
        feature of image
        """
        return self.embedding_net(x)


class ContrastiveLoss(nn.Module):
    """
    The contractive loss loss function compares a pair of images.
    According to the prior information of image pairs provided by csv,
    similar images can be arranged closer, and dissimilar images can be arranged farther.
    """

    def __init__(self, margin):
        """

        Parameters
        ----------
        margin :
        """
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.eps = 1e-9

    def forward(self, output1, output2, target, size_average=True):
        distances = (output2 - output1).pow(2).sum(1)  # squared distances
        #distances = torch.cdist(output1, output2, p=2)
        losses = 0.5 * (target.float() * distances +
                        (1 + -1 * target).float() * F.relu(self.margin - (distances + self.eps).sqrt()).pow(2))
        return losses.sum()

cuda = torch.cuda.is_available()
conv_net = models.resnet50(pretrained=False, progress=True)
conv_net  = torch.nn.Sequential(*(list(conv_net.children())[:-1]))
embedding_net = EmbeddingNet(conv_net=conv_net)
# Step 3
model = SiameseNet(embedding_net)

cuda = torch.cuda.is_available()
conv_net = models.resnet50(pretrained=False, progress=True)
conv_net  = torch.nn.Sequential(*(list(conv_net.children())[:-1]))
embedding_net = EmbeddingNet(conv_net=conv_net)
# Step 3
model = SiameseNet(embedding_net)

margin = 1.
loss_fn = ContrastiveLoss(margin)
lr = 0.05
optimizer = optim.Adam(model.parameters(), lr=lr)
scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.05, last_epoch=-1)
n_epochs = 1
log_interval = 500
device = torch.device("cuda")

def load_model_continue_training(checkpoint_path, model):
    checkpoint = torch.load(checkpoint_path)
    #model.load_state_dict(checkpoint['model_state_dict'])
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    return model


# next time ONLY CHANGE PATH
model = load_model_continue_training('/home/tianyi/model.pt', model)

if cuda:
    model.cuda()

outputs = model(*data)

import torch
import numpy as np
train_losses= []
valid_losses= []
current_loss = 0

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

        message = 'Epoch: {}/{}. Train set: Average loss: {}'.format(epoch + 1, n_epochs, train_loss)
        for metric in metrics:
            message += '\t{}: {}'.format(metric.name(), metric.value())

        val_loss, metrics = test_epoch(val_loader, model, loss_fn, cuda, metrics)
        val_loss/= len(valid_loader)
        valid_losses.append(val_loss)

        message += '\nEpoch: {}/{}. Validation set: Average loss: {}'.format(epoch + 1, n_epochs, val_loss)
        for metric in metrics:
            message += '\t{}: {}'.format(metric.name(), metric.value())

        print(message)
        current_loss = val_loss


def train_epoch(train_loader, model, loss_fn, optimizer, cuda, log_interval, metrics, limit=1000000):
    for metric in metrics:
        metric.reset()

    model.train()
    losses = []
    total_loss = 0
    num_trained = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        num_trained += batch_size
        if num_trained >= limit:
            break
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
        print(f"Train Epoch: current batch is {batch_idx} with {num_trained} images.# with batch loss: {loss.item()}")

        for metric in metrics:
            metric(outputs, target, loss_outputs)

        if batch_idx % log_interval == 0:
            message = 'Train: [{}/{} ({}%)]\tLoss: {}'.format(
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
            #total_loss += loss.item()
            print(f"Valid Epoch: current batch is {batch_idx} with {num_trained} images.# with batch loss: {loss.item()}")
            for metric in metrics:
                metric(outputs, target, loss_outputs)

    #total_loss /= (batch_idx + 1)
    #return total_loss, metrics
    return val_loss, metrics



import warnings
warnings.filterwarnings("ignore")
fit(train_loader, valid_loader, model, loss_fn, optimizer, scheduler, n_epochs, cuda, log_interval)

torch.save(
    {'epoch': n_epochs, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'loss': current_loss},
    'model_2.pt'
)

import pandas as pd
df = pd.DataFrame({'train_loss':train_losses, 'valid_loss':valid_losses})
df.to_csv('loss_2.csv')

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

final_dataset.plot(51)

