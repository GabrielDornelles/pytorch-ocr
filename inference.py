import torch
import albumentations
import numpy as np
from PIL import Image 

from models.resnet_gru import ResNetGRU
from utils.ctc_decoder import decode_predictions


# I use "∅" to denote the blank token. This list is automatically generated at training, 
# but I recommend that you hardcode your characters at evaluation
classes = ['∅','1', '2', '3', '4', '5', '6', '7', '8', '9', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']

def inference(image_path):
    # Hardcoded resize
    image = Image.open(image_path).convert("RGB")
    image = image.resize((180, 50), resample=Image.BILINEAR)
    image = np.array(image)

    # ImageNet mean and std (not required, but if you trained with, keep it)
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    aug = albumentations.Compose([
            albumentations.Normalize(mean, std, max_pixel_value=255.0, always_apply=True)
        ])

    image = aug(image=image)["image"]
    image = np.transpose(image, (2, 0, 1)).astype(np.float32)
    image = image[None,...] 
    image = torch.from_numpy(image)
    if str(device) == "cuda": image = image.cuda()
    image = image.float()
    with torch.no_grad():
        preds, _ = model(image)
    answer = decode_predictions(preds, classes)
    return answer

if __name__ == "__main__":
    # Setup model and load weights
    model = ResNetGRU(35)
    device = torch.device("cuda")
    model.to(device)
    model.load_state_dict(torch.load("./logs/crnn_best.pth"))
    model.eval()

    filepath = "/home/gabriel/Documents/Gabriel/TorchNN-OCR/cnpj_captchas/1sk9jj.png"
    answer = inference(filepath)
    print(f"text: {answer}")
