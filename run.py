from models import SimilarityNet
from torch import LongTensor, rand, randn

dev = "cuda"
# x = randn((1, 3, 312, 312)).to(dev)
# y = rand((1, 440)).type(LongTensor).to(dev)

x = randn((1, 3, 312, 312))
y = rand((1, 440)).type(LongTensor)

bcn = SimilarityNet()
res = bcn(x,y)
print(res)

