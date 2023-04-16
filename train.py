import torch
import torch.nn.functional as F
from models import SimilarityNet
import torch.nn as nn
# from prepare_dataset import train_dataloader
from prepare_dataset import dataset
import os
from config import model_path, batch_size
from torch.utils.data import DataLoader
from random import randint
from models import ContrastiveLoss
from torch.utils.data import Subset


# train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
# print(train_dataloader.dataset.vocab_size)

device = "cuda" if torch.cuda.is_available() else "cpu"


def model_eval(model, dataset = dataset):
    # dataloader must be the held out test set
    indices = []
    correct, ns = 0, 12
    for _ in range(ns):
        indices.append(randint(0, len(dataset)-1))

    subloader = Subset(dataset, indices)

    model.eval()
    ns = 10
    with torch.no_grad():
        for img, txt, y in subloader:
            img, txt = img.unsqueeze(dim=0), txt.unsqueeze(dim=0)
            img, txt = img.to(device), txt.to(device)
            # print(img.shape, txt.shape)
            _img, _txt = model(img, txt)
            score = F.pairwise_distance(_img, _txt).item()
            print(f"similarity idx => {score}; gt => {y.item()}")
            
            
            



def train_model(model, criterion, optimizer, epochs, save_model=True):

    model.to(device)
    print(f"[\] model initialized on {device}")
    print("[\] Training ...")
    train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    for _ in range(epochs):
        # train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        for step, (img, txt, y) in enumerate(train_dataloader):
            
            img, txt, y = img.to(device), txt.to(device), y.to(device)
            # print(img.shape, txt.shape)
            optimizer.zero_grad()
            img_, txt_ = model(img, txt)
            # print(img_.shape, txt_.shape)
            # print(preds, preds.shape)
            # print(y, y.shape)
            # print(preds, y)
            loss = criterion(img_, txt_, y)

            loss.backward()
            optimizer.step()
            
            if step%1 == 0:
                print(f"epoch => {_}; loss => {loss}; step => {step}; gt_ => {y.item()}")    
        print("[/] Evaluating model ...")
        model_eval(model, dataset)
        print()

    if save_model:
        os.makedirs("./models", exist_ok=True)
        torch.save(model.state_dict(), model_path)
        print("Saved Model !")

if __name__ == "__main__":
    epochs = 50
    model = SimilarityNet()
    # criterion = nn.BCELoss() # output, target is the form of parameters
    # criterion = nn.TripletMarginLoss(margin=1.0, p=2)
    criterion = ContrastiveLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = 3e-4)
    
    train_model(model, criterion, optimizer, epochs, True)

