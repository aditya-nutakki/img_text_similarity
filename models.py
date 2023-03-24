import torch.nn as nn
import torch.functional as F


class ImageModel(nn.Module):
    def __init__(self, input_shape = 256):
        super().__init__()
        self.input_shape = input_shape
        # assuming standard rgb image
        self.conv = nn.Conv2d(3, 64, 3)
        self.conv1 = nn.Conv2d(64, 128, 3, stride=2)
        self.conv2 = nn.Conv2d(128, 128, 3, stride=2)
        self.dropout = nn.Dropout(0.25)
        self.batchnorm1 = nn.BatchNorm2d(64)
        self.batchnorm2 = nn.BatchNorm2d(128)

    
    def forward(self, x):
        print(x.shape)
        x = self.conv(x)
        print(x.shape)
        x = self.dropout(x)
        print(x.shape)
        x = self.batchnorm1(x)
        print(x.shape)

        x = self.conv1(x)
        print(x.shape)
        x = self.dropout(x)
        x = self.batchnorm2(x)
        print(x.shape)
        x = self.conv2(x)
        print(x.shape)
        print(f"x was here")

        return x


class LanguageModel(nn.Module):
    def __init__(self, input_shape = 128) -> None:
        super().__init__()
        self.input_shape = input_shape


class BCNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.language_model = LanguageModel()
        self.image_model = ImageModel()


