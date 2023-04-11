from models import SimilarityNet
from torch import LongTensor, rand, randn
import torch
import torch.nn as nn

dev = "cuda"
# x = randn((1, 3, 312, 312)).to(dev)
# y = rand((1, 440)).type(LongTensor).to(dev)

imgs = randn((4, 3, 256, 256))
txts = rand((4, 440)).type(LongTensor)

bcn = SimilarityNet()
res = bcn(imgs, txts)
print(res, res.shape)

y = torch.Tensor([[1], [0], [1], [1]])
criterion = nn.BCELoss()

print(f"loss => {criterion(res, y)}")

