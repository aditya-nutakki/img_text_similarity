import torch, torchvision
from torch.utils.data import DataLoader, Subset, Dataset
from random import randint, choice
import torch.nn as nn
from torchvision import transforms
import os, json
import torch.nn.functional as F
import cv2
import sys
import time


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


class SiameseNet(nn.Module):
    def __init__(self, dataset_path = "/mnt/d/work/datasets/colors/classes"):
        super().__init__()
        self.dataset_path = dataset_path
        self.nc = len(os.listdir(dataset_path))
        
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, 5, stride=2),
            nn.ReLU(),
            nn.Conv2d(16, 3, 5, stride=4),
            nn.ReLU()
        )
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(2883, self.nc)

    def forward_once(self, x):
        x = self.conv(x)
        # print(x.shape)
        x = self.flatten(x)
        # print(x.shape)
        x = F.relu(self.linear(x))
        # print(x, x.shape)
        return x

    def forward(self, img1, img2):
        return self.forward_once(img1), self.forward_once(img2)


# ------------------------------------
class SiameseDataset(Dataset):
    def __init__(self, img_size = 256, dataset_path = "/mnt/d/work/datasets/colors/classes"):
        super().__init__()
        self.img_size = img_size
        self.transforms = transforms.Compose(
            [transforms.ToTensor(), 
             transforms.Resize((self.img_size, self.img_size))]
        )
        self.dataset_path = dataset_path
        self.nc = len(os.listdir(dataset_path))
        self.all_imgs, self.img_count = self.get_dataset_info(self.dataset_path)
    
    def get_dataset_info(self, dataset_path):
        imgs = []
        classes = os.listdir(dataset_path)
        for class_ in classes:
            cur_path = os.path.join(dataset_path, class_)
            for _img in os.listdir(cur_path):
                imgs.append(os.path.join(cur_path, _img))
        return imgs, len(imgs)

    def get_pairs(self, idx):
        img1_path = self.all_imgs[idx]
        img2_path = choice(self.all_imgs)

        img1_class = img1_path.split("/")[-2]
        img2_class = img2_path.split("/")[-2]
        # pos_pair, neg_pair = [], []

        if img1_class == img2_class:
            pos_pair = [img1_path, img2_path, 0]
            while img1_class == img2_class:
                img2_path = choice(self.all_imgs)
                img2_class = img2_path.split("/")[-2]

            neg_pair = [img1_path, img2_path, 1]

        elif img1_class != img2_class:
            neg_pair = [img1_path, img2_path, 1]
            while img1_class != img2_class:
                img2_path = choice(self.all_imgs)
                img2_class = img2_path.split("/")[-2]

            pos_pair = [img1_path, img2_path, 0]

        return pos_pair, neg_pair
            

    def __len__(self):
        return self.img_count

    def __getitem__(self, index):
        pos_pair, neg_pair = self.get_pairs(index)
        rand_idx = randint(0, 1)
        if rand_idx:
            pair = pos_pair
        else:
            pair = neg_pair
        
        img1_path, img2_path, label = pair
        print(img1_path, img2_path, label)
        label = torch.Tensor([label])
        img1, img2 = self.transforms(cv2.imread(img1_path)), self.transforms(cv2.imread(img2_path))
        return img1, img2, label.unsqueeze(dim=0)


# model = SiameseNet()
# x = torch.randn(1, 3, 256,256)
# y = torch.randn(1, 3, 256,256)
# model(x,y)


def load_model(model_path):
    model = SiameseNet()
    model.load_state_dict(torch.load(model_path))
    return model


# ds = SiameseDataset()
# # img1, img2, label = ds[12]
# dataloader = DataLoader(ds, batch_size=4, shuffle=True)


def train(model, criterion, opt, epochs, save_model=True):
    ds = SiameseDataset()
    # # img1, img2, label = ds[12]
    dataloader = DataLoader(ds, batch_size=4, shuffle=True)

    for ep in range(epochs):
        for step, (img1, img2, label) in enumerate(dataloader):
            opt.zero_grad()
            _img1, _img2 = model(img1, img2)
            loss = criterion(_img1, _img2, label)
            print(label)
            loss.backward()
            opt.step()

            if step%2 == 0:
                print(f"epoch => {ep}; loss => {loss}; step => {step}")
        print()

    if save_model:
        os.makedirs("./models", exist_ok=True)
        torch.save(model.state_dict(), "./models/color_siamese.pt")
        print("model saved !")

if __name__ == "__main__":
    mode = sys.argv[1]

    if mode == "train":
        epochs = 15
        model = SiameseNet()
        criterion = ContrastiveLoss()
        opt = torch.optim.Adam(model.parameters(), lr = 3e-4)
        train(model, criterion, opt, epochs)

    else:
        # mode = "inference"
        trnsfrms = transforms.Compose(
            [transforms.ToTensor(), transforms.Resize((256,256))]
        )
        img1 = trnsfrms(cv2.imread("/mnt/d/work/datasets/colors/classes/red/6.jpg")).unsqueeze(dim=0)
        img2 = trnsfrms(cv2.imread("/mnt/d/work/datasets/colors/classes/white/57.jpg")).unsqueeze(dim=0)

        print(img1.shape, img2.shape)
        stime = time.time()
        model = load_model("./models/color_siamese.pt")
        ftime = time.time()

        print(f"[/] Loaded Model in {ftime-stime}s")
        model.eval()
        with torch.no_grad():
            stime = time.time()
            _img1, _img2 = model(img1, img2)
            score = F.pairwise_distance(_img1, _img2, keepdim=True).item()
            ftime = time.time()
            print(f"Similarity Score => {score}; inference done in {ftime-stime}")





