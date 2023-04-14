import torch
from models import SimilarityNet
import torch.nn as nn
from prepare_dataset import train_dataloader
import os
from config import model_path

print(train_dataloader.dataset.vocab_size)


def train_model(model, criterion, optimizer, epochs, save_model=True):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # model.to(device)
    print(f"[\] model initialized on {device}")
    print("[\] Training ...")
    for _ in range(epochs):
        for step, (img, txt, y) in enumerate(train_dataloader):
            
            # img, txt, y = img.to(device), txt.to(device), y.to(device)
            print(img.shape, txt.shape)
            optimizer.zero_grad()
            preds = model(img, txt)
            print(preds, preds.shape)
            print(y, y.shape)
            loss = criterion(preds, y)

            loss.backward()
            optimizer.step()
            print(f"epoch => {_}; loss => {loss}; step => {step}")
        print()
    
    if save_model:
        os.makedirs("./models", exist_ok=True)
        torch.save(model.state_dict(), model_path)
        print("Saved Model !")

if __name__ == "__main__":
    epochs = 1
    model = SimilarityNet()
    criterion = nn.BCELoss() # output, target is the form of parameters
    optimizer = torch.optim.Adam(model.parameters(), lr = 3e-4)
    
    train_model(model, criterion, optimizer, epochs, False)

