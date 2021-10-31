from jina import DocumentArray, Executor, requests

import torch
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms

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


def load_model_continue_training(checkpoint_path, model):
    device = torch.device("cpu")
    lr = 0.05
    optimizer = optim.Adam(model.parameters(), lr=lr)
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    #model.load_state_dict(checkpoint['model_state_dict'])
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    return model


class LandmarkEncoder(Executor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        conv_net = models.resnet50(pretrained=False, progress=True)
        conv_net = torch.nn.Sequential(*(list(conv_net.children())[:-1]))
        embedding_net = EmbeddingNet(conv_net=conv_net)
        # Step 3
        model = SiameseNet(embedding_net)
        self.my_model = load_model_continue_training('/Users/mac/Documents/code/project/a/landmark/executor/encode/model.pt', model)
        #self.my_model = torch.load('G:/project/a/landmark/executor/encode/model.pt', map_location=torch.device('cpu'))
        self.my_model.embedding_net.fc[4] = nn.Identity()
        self.my_model.embedding_net.fc[5] = nn.Identity()
        self.my_model.embedding_net.fc[6] = nn.Identity()
        self.my_model.embedding_net.fc[7] = nn.Identity()

        self.my_transformer = transforms.Compose([
            transforms.ToTensor(),  # 抄 将输入值转成tensor
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),  # 抄
        ])


    @requests 
    def process(self, docs: DocumentArray, **kwargs):
        for doc in docs:
            blob = self.my_transformer(doc.blob) 
            blob= torch.unsqueeze(blob, 0)
            embedding = self.my_model.get_embedding(blob)
            doc.embedding= embedding.detach().numpy()[0]



