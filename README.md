# TorchNN-OCR



A PyTorch simple framework to train Optical Character Recognition (OCR) models. 

You can train models to read **captchas**, **license plates**, **digital displays**, and any type of text!

See:

![image](https://user-images.githubusercontent.com/56324869/206953640-087d17b1-a0a7-4f99-ad82-d8c93365bd41.png)


# Rich Text while Training!

![image](https://user-images.githubusercontent.com/56324869/206952565-1da49dc0-d3ee-4328-8855-19f62aafb435.png)

# Hydra Logs!
You have the whole **Training Log** in a train.log file so you can process it anywhere!

![image](https://user-images.githubusercontent.com/56324869/207184241-855019e3-889d-4c2d-ae11-62dd73f62352.png)


# How to train?


- Create a directory called "dataset" and throw your images there (preferable to be png, but you can use other formats as long as you change [that](https://github.com/GabrielDornelles/EchidNet-OCR/blob/5275b1169051763fbb08f583871a28e88c706454/train.py#L56))

- Your file tree should be like that:
    ```
    torch-nn-ocr
    │   README.md
    │   ...  
    │
    └─── dataset
        cute.png
        motor.png
        machine.png
    ```
    The image name needs to be the content writen in the image. In this case you have one image with 'cute' written in it, other with 'motor' and another with 'machine'.

- Your data should be of same length, padding is done automatically if using Attention + CrossEntropy, but padding is not done for CTC Loss, so make sure you normalize your target lengths in case of using CTC Loss.

- Run:
```sh
python3 train.py
```
## Currently Support:

- CRNNs ✅
- Attention + CTC Loss✅ 
- Attention + Cross Entropy Loss ✅

## Will Support:
- Other backbones

# TODO:
- ~~Add logging with hydra, so it saves logging in text files~~. ✅
- Add CI with github actions, to test if everything works fine after pushes to this repo.
- Add tests to main methods so it keeps secure when adding more models and functionalities in the future.