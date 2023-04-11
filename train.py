import torch
from models import SimilarityNet
import torch.nn as nn


def train_model(model, criterion, optimizer, epochs, save_model=True):
    
    for _ in range(epochs):
        for step, (img, txt, y) in enumerate(training_dataloader):
            optimizer.zero_grad()
        
            preds = model(img, txt)
            loss = criterion(preds, y)

            loss.backward()
            optimizer.step()
            

if __name__ == "__main__":
    epochs = 10
    model = SimilarityNet()
    criterion = nn.BCELoss() # output, target is the form of parameters
    optimizer = torch.optim.Adam(model.parameters(), lr = 3e-4)
    
    train_model(model, criterion, optimizer, epochs, True)

