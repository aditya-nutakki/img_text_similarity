import os
import torch
from torch.utils.data import Dataset, DataLoader
from utils.nlp_utils import create_vocab_inv_labels, text2int, int2text
import time


base = "/mnt/d/work/datasets/img_text"
labels_path = os.path.join(base, "labels/")
images_path = os.path.join(base, "images/")

labels, vocab, inv_vocab = create_vocab_inv_labels(labels_path)

