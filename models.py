import torch.nn as nn
import torch.nn.functional as F
import torch
from config import padding_length, img_size, vocab_path
import json

class ImageModel(nn.Module):
    def __init__(self, input_shape = img_size):
        super().__init__()
        self.input_shape = input_shape
        # assuming standard rgb image
        self.conv = nn.Conv2d(3, 128, 5, stride=2)
        self.conv1 = nn.Conv2d(128, 256, 5, stride=2)
        self.conv2 = nn.Conv2d(256, 64, 5, stride=2)
        self.conv3 = nn.Conv2d(64, 16, 5, stride=3)
        self.dropout = nn.Dropout(0.25)
        self.batchnorm1 = nn.BatchNorm2d(128)
        self.batchnorm2 = nn.BatchNorm2d(256)

        self.flatten = nn.Flatten()
    
    def forward(self, x):
        # print(x.shape)
        x = F.relu(self.conv(x))
        x = self.dropout(x)
        x = self.batchnorm1(x)
        # print(x.shape)
        x = self.conv1(x)
        # print(x.shape)
        x = self.dropout(x)
        x = self.batchnorm2(x)
        x = F.relu(self.conv2(x))
        # print(x.shape)
        x = F.relu(self.conv3(x))
        # print(x.shape)
        x = self.flatten(x)
        # print(f"image op => {x.shape}")n 
        return x


class LanguageModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding_dim = 1024
        self.vocab_size = len(json.load(open(vocab_path))) + 1
        self.embedding = nn.Embedding(self.vocab_size, self.embedding_dim)
        self.flatten = nn.Flatten()
        # self.conv = nn.Conv1d(self.vocab_size, 8, kernel_size=5, stride=2)
        self.conv = nn.Conv2d(1, 32, kernel_size=7, stride=4)
        self.lstm = nn.LSTM(self.embedding_dim, 256, 2, batch_first = True, dropout = 0.15)
        
    def forward(self, x):
        # print(x.shape, self.vocab_size)
        x = self.embedding(x)
        # print(x.shape)
        # x = F.relu(self.conv(x))
        _, (x, _) = self.lstm(x)
        x = self.flatten(x)
        # print(f"nlp op => {x.shape}")
        return x


class SimilarityNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.language_model = LanguageModel()
        self.image_model = ImageModel()
        self.linear = nn.Linear(256, 96)
        self.linear2 = nn.Linear(96, 32)
        self.linear3 = nn.Linear(32, 1)

        # self.bilinear = nn.Bilinear(1296, 2032, 128)
        self.bilinear = nn.Bilinear(1296, 512, 256)
        # is of the form (img, txt, output_)

    def forward(self, img, txt):
        txt = self.language_model(txt)
        img = self.image_model(img)

        cat = self.bilinear(img, txt)
        # print(cat.shape)
        # cat = torch.cat((img, txt), dim=1)
        cat = F.relu(self.linear(cat))
        cat = F.relu(self.linear2(cat))
        cat = F.sigmoid(self.linear3(cat))
        return cat

    