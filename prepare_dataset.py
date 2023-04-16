import os
import torch
from torch.utils.data import Dataset, DataLoader
from utils.nlp_utils import create_vocab_inv_labels, text2int, int2text
import time
from torchvision import transforms
from config import padding_length, img_size, batch_size
import cv2
from random import randint
import numpy as np


base = "/mnt/d/work/datasets/colors"
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
        # print(f"raw text => {text} is of type {type(text)}")
        # make a random guess whether to give label pos or neg; if neg generate a different label than what it is 
        gt_label = text[0][0]

        rand_label = randint(0,1)
        # label_ = torch.Tensor([1])
        if rand_label:
            # label_ = torch.Tensor([1])
            label_ = torch.Tensor([0])
        else:
            gen_label = randint(1,6)
            while gen_label == gt_label:
                gen_label = randint(1,6)

            text = [[gen_label, 0, 0, 0]]
            # label_ = torch.Tensor([0])
            label_ = torch.Tensor([1])
        

        img_path = self.images_path + img_path.split("/")[-1].replace(".txt", ".jpg")
        img = self.transforms(cv2.imread(img_path))

        # returns text, image, label
        # return img, torch.Tensor(text).type(torch.LongTensor), torch.Tensor([1])
        return img, torch.Tensor(text).type(torch.LongTensor), label_

dataset = ImgTextDataset(img_text_pairs=img_text_pairs)
# print(dataset[1])
# train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


if __name__ == "__main__":
    dummy_sp = "./dummy/"
    os.makedirs(dummy_sp, exist_ok=True)

    while True:
        idx = int(input())
        img, txt, label = dataset[idx]
        img = img.detach().cpu().numpy()
        txt = txt.detach().cpu().numpy()
        label = label.detach().cpu().numpy()
        # converting (c, h, w) to (w, h, c)
        img = img.transpose((1,2,0))
        print(txt, label)
        print(img.shape, txt.shape, label.shape)
        label_ = str(txt)[9]
        save_path = os.path.join(dummy_sp, f"{idx}_{txt[0][0]}_{int(label[0])}") + ".jpg"
        cv2.imwrite(save_path, img*255)
        print()
        # print(len(dataset))