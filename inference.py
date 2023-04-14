import torch
from config import model_path, img_size, vocab_path, padding_length
from models import SimilarityNet
import cv2
from torchvision import transforms
from utils.nlp_utils import preprocess_raw_text, pad
import json, time

trnsfrms = transforms.Compose([
    transforms.ToTensor(), 
    transforms.Resize((img_size, img_size))
])

def load_model():
    model = SimilarityNet()
    model.load_state_dict(torch.load(model_path))
    return model


def text2tensor(raw_text):
    def _numericalize(rt):
        transformed_text = []
        for word in rt:
            if word not in vocab:
                word = "unk"
            transformed_text.append(vocab[word])
        
        if len(transformed_text) < padding_length:
            transformed_text = pad(transformed_text)
        return transformed_text
    
    vocab = json.load(open(vocab_path))
    # raw_text = text2int(raw_text)
    raw_text = preprocess_raw_text(raw_text)
    raw_text = pad(_numericalize(raw_text))
    
    return torch.Tensor(raw_text).type(torch.LongTensor).reshape((1,1,-1))


def preprocess_input(img_path, txt):
    img = trnsfrms(cv2.imread(img_path)).unsqueeze(dim=0)
    txt = text2tensor(txt)
    # print(img.shape, txt.shape)
    return img, txt



img_path = "/mnt/d/work/datasets/img_text/117.jpg"
txt = "buildings buildings buildings buildings buildings buildings"
img, txt = preprocess_input(img_path, txt)


model = load_model()
model.eval()
print()
stime = time.time()
with torch.no_grad():
    preds = model(img, txt)
    print(preds, preds.shape)
ftime = time.time()

print(f"evaluation done in {ftime - stime}s")

