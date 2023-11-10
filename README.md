# TorchNN-OCR

A simple PyTorch framework to train Optical Character Recognition (OCR) models. 

You can train models to read **captchas**, **license plates**, **digital displays**, and any type of text!

See:

<p align="center">
  <img src="https://user-images.githubusercontent.com/56324869/206953640-087d17b1-a0a7-4f99-ad82-d8c93365bd41.png" />
</p>


# Rich Text while Training!

<p align="center">
  <img src="https://user-images.githubusercontent.com/56324869/206952565-1da49dc0-d3ee-4328-8855-19f62aafb435.png" />
</p>

# Pre-requisites

1. if you're using Windows OS, install [CUDA](https://developer.nvidia.com/cuda-toolkit-archive) according to Python, [Pytorch](https://pytorch.org/get-started/previous-versions/) and [Torch vision](https://pypi.org/project/torchvision/) versions compliance.
  - e.g. python==3.10.13 pytorch==1.12.1 torchvision==0.13.1 cudatoolkit=11.6
2. install the rest of dependencies: `pip install -r requirements.txt`
  - if you encounter issues with numpy package import, just reinstall it; it's a known issue

# Hydra!
You have the whole **Training Log** in a train.log file so you can process it anywhere!

<p align="center">
  <img src="https://user-images.githubusercontent.com/56324869/207184241-855019e3-889d-4c2d-ae11-62dd73f62352.png"/>
</p>


You can also run multiple training runs with Hydra:
```sh
python3 train.py --multirun model.use_attention=true,false model.use_ctc=true,false training.num_epochs=50,100
```

This example will run 8 different trainings with each configuration.

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

- Your data should be of same length, padding is done automatically if using Attention + CrossEntropy, but padding is not done for CTC Loss, so make sure you normalize your target lengths in case of using CTC Loss (you can do this by adding a character to represent empty space, remember to not use the same as CTC uses for blank, those are different blanks).

- Configure your model at ```configs/config.yaml```
  ```yaml
  model:
    use_attention: true 
    use_ctc: true
    dims: 256
  ```
- Run:
```sh
python3 train.py
```
## Support:

- CRNNs ✅
- Attention ✅
- CTC Loss ✅ 
- Cross Entropy Loss ✅

## Will Support:
- Other backbones
- Text detection models(would you like it?)

# TODO:
- ~~Add logging with hydra, so it saves logging in text files~~. ✅
- ~~Add CI with github actions, to test if everything works fine after pushes to this repo~~. ✅
- ~~Add tests to main methods so it keeps secure when adding more models and functionalities in the future.~~ Partially added. ✅
- Configure Dockerfile for inference