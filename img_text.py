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
from config import vocab_path, padding_length

class ContrastiveLoss(torch.nn.Module):
    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
      # Calculate the euclidian distance and calculate the contrastive loss
        euclidean_distance = F.pairwise_distance(output1, output2, keepdim = True)
        print(f"similarity scores => {euclidean_distance}")
        loss_contrastive = torch.mean((1-label) * torch.pow(euclidean_distance, 2) +
                                    (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))

        return loss_contrastive


class SiameseNet(nn.Module):
    def __init__(self, dataset_path = "/mnt/d/work/datasets/colors/classes"):
        super().__init__()
        self.dataset_path = dataset_path
        self.nc = len(os.listdir(dataset_path))
        
        # input image res = (3, 72, 72)
        self.img_net = nn.Sequential(
            nn.Conv2d(3, 16, 5, stride=2),
            nn.ReLU(),
            nn.Conv2d(16, 3, 5, stride=4),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(192, 32),
            nn.ReLU(),
            nn.Linear(32, 8),
            nn.ReLU()
        )
        self.linear_img = nn.Linear(192, 32)
        self.linear_img2 = nn.Linear(32, 8)
        


        self.vocab_size = len(json.load(open(vocab_path))) + 1
        self.embedding_dim = 8
        self.nlp_net = nn.Sequential(
            nn.Embedding(self.vocab_size, self.embedding_dim),
            nn.Flatten(),
            nn.Linear(32,8),
            nn.ReLU()
        )
        self.flatten = nn.Flatten()
    
    def forward(self, img, txt):
        return self.img_net(img), self.nlp_net(txt)

# ------------------------------------
class SiameseDataset(Dataset):
    def __init__(self, img_size = 72, dataset_path = "/mnt/d/work/datasets/colors/classes"):
        super().__init__()
        self.img_size = img_size
        self.transforms = transforms.Compose(
            [transforms.ToTensor(), 
             transforms.Resize((self.img_size, self.img_size))]
        )
        self.dataset_path = dataset_path
        self.nc = len(os.listdir(dataset_path))
        self.all_imgs, self.img_count = self.get_dataset_info(self.dataset_path)
        self.vocab = json.load(open(vocab_path))


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
            pos_pair = [img1_path, img2_class, 0]
            while img1_class == img2_class:
                img2_path = choice(self.all_imgs)
                img2_class = img2_path.split("/")[-2]

            neg_pair = [img1_path, img2_class, 1]

        elif img1_class != img2_class:
            neg_pair = [img1_path, img2_class, 1]
            while img1_class != img2_class:
                img2_path = choice(self.all_imgs)
                img2_class = img2_path.split("/")[-2]

            pos_pair = [img1_path, img2_class, 0]

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
        
        img_path, txt, label = pair
        # print(img1_path, img2_path, label)
        label = torch.Tensor([label])
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img, txt = self.transforms(img), torch.Tensor([self.vocab[txt], 0, 0, 0]).type(torch.LongTensor)
        return img, txt, label.unsqueeze(dim=0)


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
    # img, txt, label = ds[4]
    # print(img, txt, label)
    # print(img.shape, txt.shape, label.shape)
    dataloader = DataLoader(ds, batch_size=4, shuffle=True)

    for ep in range(epochs):
        for step, (img, txt, label) in enumerate(dataloader):
            opt.zero_grad()
            # print(img.shape, txt.shape, label.shape)
            _img, _txt = model(img, txt)
            # print(_img.shape, _txt.shape, label.shape)
            loss = criterion(_img, _txt, label)
            print(label)
            print()
            loss.backward()
            opt.step()

            if step%2 == 0:
                print(f"epoch => {ep}; loss => {loss}; step => {step}")
        print()

    if save_model:
        os.makedirs("./models", exist_ok=True)
        torch.save(model.state_dict(), "./models/color_siamese2.pt")
        print("model saved !")

if __name__ == "__main__":
    mode = sys.argv[1]

    if mode == "train":
        epochs = 15
        model = SiameseNet()
        criterion = ContrastiveLoss()
        opt = torch.optim.Adam(model.parameters(), lr = 3e-4)
        train(model, criterion, opt, epochs)

    elif mode == "infer":
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

    else:
        # mode for testing
        model = SiameseNet()
        # batched 4d input for images, batched 2d input for txt
        x = torch.randn(1, 3, 72, 72)
        y = torch.Tensor([[4,0,0,0]]).type(torch.LongTensor)
        print(y.shape)
        x, y = model(x, y)
        print(x.shape, y.shape)
        print(x, y)


