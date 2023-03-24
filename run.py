from models import ImageModel
import torch

x = torch.randn((1, 3,256,256))
img_model = ImageModel()
img_model(x)
