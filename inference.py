import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from torch.nn import functional as F
import os
from sklearn import preprocessing
import time
import albumentations
from model import OcrModel
import config

model = OcrModel(33)
device = torch.device(config.DEVICE)
model.to(device)
model.load_state_dict(torch.load(config.SAVE_MODEL_AS))

def remove_duplicates(s):
    chars = list(s)
    prev = None
    k = 0
 
    for c in s:
        if prev != c:
            chars[k] = c
            prev = c
            k = k + 1
    
    return ''.join(chars[:k])

def return_model_answer(img_path):
    """
    input: image_path
    output: character on the image classified
    """
    ## transforming image into tensor with right shape
    model.eval()
    start = time.time()

    img = img_path
    image = Image.open(img_path).convert("RGB")
    image = image.resize(
                (config.IMAGE_WIDTH, config.IMAGE_HEIGHT), resample=Image.BILINEAR
            )
    image = np.array(image)

    # Some ImageNet data tricks
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    aug = albumentations.Compose(
        [
            albumentations.Normalize(
                mean, std, max_pixel_value=255.0, always_apply=True
            )
        ]
    )
    image = aug(image=image)["image"]
    image = np.transpose(image, (2, 0, 1)).astype(np.float32)
    image = image[None,:,:,:] #add batch dimension as 1
    img = torch.from_numpy(image)
    if str(device) == "cuda":
        img= img.cuda()
    img = img.float()
    
    preds, _ = model(img) # tensor with each class output from net

    preds = preds.permute(1, 0, 2)
    preds = torch.softmax(preds, 2)
    preds = torch.argmax(preds, 2)
    preds = preds.detach().cpu().numpy()
  
    print(f"predicted softmax/argmax from tensor: {preds}")
    captcha = preds[0]
    decode = {
    0:"",
    1:"1",
    2:"2",
    3:"3",
    4:"4",
    5:"5",
    6:"6",
    7:"7",
    8:"8",
    9:"a",
    10:"b",
    11:"c",
    12:"d",
    13:"e",
    14:"f",
    15:"g",
    16:"h",
    17:"i",
    18:"j",
    19:"k",
    20:"l",
    21:"m",
    22:"n",
    23:"o",
    24:"p",
    25:"q",
    26:"r",
    27:"s",
    28:"t",
    29:"u",
    30:"v",
    31:"w",
    32:"x",
    33:"y",
    }
    answer= ""
    for i in captcha:
        answer += decode[i]

    answer = remove_duplicates(answer)
    end = time.time()
    print(answer)
    print(f"{(end - start)*1000:.2f}ms")
    print("\n")
    return answer

rootdir = 'dataset/'
right = 0
cnt = 0
for subdir, dirs, files in os.walk(rootdir):
    for file in files:
        cnt+=1
        answer = file[:5]
        input_data = return_model_answer(f"dataset/{file}")
        if input_data == answer:
            right+=1
        
print(f"right captchas {right} in {cnt}")
print(f"accuracy:{(right/cnt)*100:.2f}%")