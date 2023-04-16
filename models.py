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
        self.conv = nn.Conv2d(3, 64, 5, stride=2)
        self.conv1 = nn.Conv2d(64, 128, 5, stride=2)
        self.conv2 = nn.Conv2d(128, 64, 5, stride=2)
        self.conv3 = nn.Conv2d(64, 16, 5, stride=3)
        self.dropout = nn.Dropout(0.25)
        self.batchnorm1 = nn.BatchNorm2d(64)
        self.batchnorm2 = nn.BatchNorm2d(128)

        self.flatten = nn.Flatten(start_dim=1)
        self.linear = nn.Linear(1296, 32)
    
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
        # print(f"image op pre flatten => {x.shape}")
        x = self.flatten(x)
        x = self.linear(x)
        # print(f"image op => {x.shape}")
        return x


class LanguageModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding_dim = 8
        self.vocab_size = len(json.load(open(vocab_path))) + 1
        self.embedding = nn.Embedding(self.vocab_size, self.embedding_dim)
        self.flatten = nn.Flatten(start_dim=1)
        # self.conv = nn.Conv1d(self.vocab_size, 8, kernel_size=5, stride=2)
        self.conv = nn.Conv2d(1, 16, kernel_size=3, stride=1)
        self.lstm = nn.LSTM(self.embedding_dim, 128, 2, batch_first=True, dropout = 0.15)

    def forward(self, x):
        # print(x.shape, self.vocab_size)
        x = self.embedding(x)
        # x = F.relu(self.conv(x))
        # _, (x, _) = self.lstm(x)
        # print(f"nlp op pre flattening => {x.shape}")
        x = self.flatten(x)
        # print(f"nlp op => {x.shape}")
        return x


class SimilarityNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.language_model = LanguageModel()
        self.image_model = ImageModel()
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(128, 64)
        self.linear2 = nn.Linear(64, 32)
        self.linear3 = nn.Linear(32, 1)

        # self.bilinear = nn.Bilinear(1296, 2032, 128)
        self.bilinear = nn.Bilinear(1296, 32, 128)
        # is of the form (img, txt, output_)

    def forward(self, img, txt):
        txt = self.language_model(txt)
        img = self.image_model(img)
        
        return img, txt
        # cat = self.bilinear(img, txt)
        # # print(cat.shape)
        # cat = self.flatten(cat)
        # # print(cat.shape)
        # # cat = torch.cat((img, txt), dim=1)
        # cat = F.relu(self.linear(cat))
        # cat = F.relu(self.linear2(cat))
        # cat = F.sigmoid(self.linear3(cat))
        # return cat

    
class ContrastiveLoss(torch.nn.Module):
    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
      # Calculate the euclidian distance and calculate the contrastive loss
        euclidean_distance = F.pairwise_distance(output1, output2, keepdim = True)
        print(euclidean_distance)
        loss_contrastive = torch.mean((1-label) * torch.pow(euclidean_distance, 2) +
                                    (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))

        return loss_contrastive