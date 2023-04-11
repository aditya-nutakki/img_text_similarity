import os
import torch
from torch.utils.data import Dataset, DataLoader
from utils.nlp_utils import preprocess_label, read_txt, make_vocab, _inverse_vocab, text2int, int2text, _numericalize
import time


base = "/mnt/d/work/datasets/img_text"
labels_path = os.path.join(base, "labels/")
images_path = os.path.join(base, "images/")


labels = os.listdir(labels_path)
for l in range(len(labels)):
    labels[l] = read_txt(os.path.join(labels_path, labels[l]))
    labels[l] = preprocess_label(labels[l])

vocab = make_vocab(labels)
for l in range(len(labels)):
    for t in range(len(labels[l])):
        # print(tok_text)
        labels[l][t] = _numericalize(labels[l][t])

# print(labels)
# print(vocab)
inverse_vocab = _inverse_vocab(vocab)
