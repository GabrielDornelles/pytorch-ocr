import torch
import albumentations
import numpy as np
from PIL import Image

from models.crnn import CRNN
from utils.model_decoders import decode_predictions, decode_padded_predictions
from torchvision import transforms

transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.ToTensor(),
])

# I use "∅" to denote the blank token. This list is automatically generated at training,
# but I recommend that you hardcode your characters at evaluation
classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'a', 'c', 'd', 'e', 'g', 'h', 'k', 'n', 'o', 'p', 'q', 's', 'u', 'v', 'x', 'y', 'z']

def inference(image_path):
    # Hardcoded resize
    image = Image.open(image_path).convert("RGB")
    image = image.resize((250, 60), resample=Image.BILINEAR)
    image = transform(image)
    image = np.array(image)

    # ImageNet mean and std (not required, but if you trained with, keep it)
    # mean = (0.485, 0.456, 0.406)
    # std = (0.229, 0.224, 0.225)
    # aug = albumentations.Compose([albumentations.Normalize(mean, std, max_pixel_value=255.0, always_apply=True)])

    # image = aug(image=image)["image"]
    # image = np.transpose(image, (2, 0, 1)).astype(np.float32)
    image = image[None, ...]
    image = torch.from_numpy(image)
    if str(device) == "cuda":
        image = image.cuda()
    image = image.float()
    with torch.no_grad():
        preds, _ = model(image)

    if model.use_ctc:
        answer = decode_predictions(preds, classes)
    else:
        answer = decode_padded_predictions(preds, classes)
    return answer


if __name__ == "__main__":
    # Setup model and load weights
    model = CRNN(
        resolution=(250, 60),
        dims=256,
        num_chars=len(classes),
        use_attention=True,
        use_ctc=True,
        grayscale=True,
    )
    device = torch.device("cuda")
    model.to(device)
    model.load_state_dict(torch.load("./logs/crnn.pth"))
    model.eval()
    filepath = "sample.png"
    answer = inference(filepath)
    print(f"text: {answer}")
