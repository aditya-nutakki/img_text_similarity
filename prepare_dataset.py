import os
import torch
from torch.utils.data import Dataset, DataLoader
from utils.nlp_utils import create_vocab_inv_labels, text2int, int2text
import time
from torchvision import transforms
from config import padding_length, img_size, batch_size
import cv2
from random import randint


# base = "/mnt/d/work/datasets/img_text"
base = "/opt/infilect/aditya/datasets/img_text"
text_path = os.path.join(base, "labels/")
images_path = os.path.join(base, "images/")

img_text_pairs, vocab, inv_vocab = create_vocab_inv_labels(text_path)
# 'texts' here is still a python list, needs to be converted into a torch.Tensor (long type)


class ImgTextDataset(Dataset):
    def __init__(self, img_text_pairs, transforms=transforms.Compose([transforms.ToTensor(), transforms.Resize((img_size, img_size))])):
        super().__init__()
        self.transforms = transforms
        self.img_size = img_size
        self.padding_len = padding_length 
        self.vocab_size = len(vocab)
        self.img_text_pairs = img_text_pairs
        self.text_path, self.images_path = text_path, images_path


    def __len__(self):
        return len(self.img_text_pairs)

    def __getitem__(self, index):
        # return tensor types for both text input and image
        pair = self.img_text_pairs[index]
        text, img_path = pair[0], pair[1]
        
        rand_idx = randint(0,1)
        label_ = torch.Tensor([int(not rand_idx)])
        text = text[rand_idx]

        img_path = self.images_path + img_path.split("/")[-1].replace(".txt", ".jpg")
        img = self.transforms(cv2.imread(img_path))

        # returns text, image, label
        # return img, torch.Tensor(text).type(torch.LongTensor), torch.Tensor([1])
        return img, torch.Tensor(text).type(torch.LongTensor), label_

dataset = ImgTextDataset(img_text_pairs=img_text_pairs)
# train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# def train()
# for i in train_dataloader:
#     text, img, label = i
    

# img, txt, label = dataset[0]
# print(txt, label)
# print(img.shape, txt.shape, label.shape)
