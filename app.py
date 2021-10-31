
from jina import Document, Flow


from landmark.executor.encode import LandmarkEncoder
# from landmark.executor.indexer import AnnIndexer


import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms
torch.cuda.empty_cache()
import boto3
import cv2
from tqdm import tqdm

os.environ['JINA_LOG_LEVEL'] = 
os.environ['AWS_ACCESS_KEY_ID'] = 
os.environ['AWS_SECRET_ACCESS_KEY'] = 

s3 = boto3.resource('')


def read_image_from_s3(key):
    bucket = s3.Bucket('masterarbeit125255aa')
    img = bucket.Object(key).get().get('Body').read()
    nparray = cv2.imdecode(np.asarray(bytearray(img)), cv2.IMREAD_COLOR)
    return nparray


def read_csv_from_s3(stage):
    if stage == 'train':
        df = pd.read_csv('')
    elif stage == 'valid':
        df = pd.read_csv('')
    elif stage == 'all':
        df = pd.read_csv('')
    else:
        raise ValueError(f'not supported stage{stage}')
    return df


class SiameseGLDV2(Dataset):
    """
    Train: For each sample creates a positive or a negative pair
    Test: For each sample creates a positive or a negative pair
    """

    def __init__(self, stage: str, inferance=True):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.df = read_csv_from_s3(stage).drop_duplicates('anchor')
        self.df.drop(self.df.filter(regex="Unname"), axis=1, inplace=True)
        print(f'shape of df is {len(self.df)}, stage is {stage}')
        self.s3 = boto3.resource('s3')
        self.s3path = 'data/train/train_compress'
        if not inferance:
            self.my_transformer = transforms.Compose([
                transforms.RandomPerspective(distortion_scale=0.5, p=0.5, fill=0),
                transforms.ColorJitter(brightness=0, contrast=0, saturation=0, hue=0),
                transforms.RandomRotation(degrees=(0, 180), expand=False, center=None, fill=0, resample=None),
                transforms.RandomCrop(size=(224, 224), padding=None, pad_if_needed=False, fill=0,
                                      padding_mode='constant'),
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
        try:  #自己加
            anchor_im = read_image_from_s3(anchor_image)
        except:
            anchor_im = []
            print(f'Error by {anchor_image}')
        return label, anchor_filen, anchor_im


dataset = SiameseGLDV2(stage='all')


def generate_index_documents(start_index=50000, end_index=50010):
    for index, (label, anchor_filen, element) in enumerate(dataset):
        if start_index < index <= end_index:
            d = Document(id=anchor_filen)
            d.blob = element
            yield d
        if index > end_index:
            break


def generate_search_document():
    for index, element in tqdm(enumerate(dataset)):
        if index == 302:
            d = Document(id=index)
            d.blob = element
            return d


f_index = Flow().add(name='step1', uses=LandmarkEncoder).add(
    name='step2',
    uses='jinahub+docker://PostgreSQLStorage/latest',
    install_requirements=True,
    uses_with={
        'hostname': 'host.docker.internal',
        'port': 5432,
        'username': 'postgres',
        'password': 'postgres',
        'database': 'HTY',
        'table': 'feature'
    },
    docker_kwargs={'environment': ["JINA_LOG_LEVEL=DEBUG"]
    }

)

# f_search = Flow(port_expose=12312, protocol='http', cors=True).add(name='step1', uses=LandmarkEncoder).add(
#     name='step2', uses=AnnIndexer, uses_with={'task': 'search'})

if __name__ == "__main__":
    # f.plot('myflow.svg')
    with f_index:  # uses_with比withYAML 中的预定义配置具有更高的优先级。当两者都出现时，uses_with首先被拿起。
        f_index.index(inputs=generate_index_documents, show_progress=True, request_size=10)
        #f_index.post(on='/dump', parameters={'dump_path': '/Users/mac/Documents/code/project/a/workspace/', 'shards': 1})
    # with f_search:
    #     search_result = f_search.search(inputs=generate_search_document, return_results=True)
    #     query = search_result[0].docs[0]
    #     print(len(query.matches))
    #     plt.imshow(query.blob)
    #     plt.show()
    #     for match in query.matches[:20]:
    #         match.blob = dataset[int(match.id)]
    #         plt.imshow(match.blob)
    #         plt.show()