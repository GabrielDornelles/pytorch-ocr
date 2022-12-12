# TorchNN-OCR



A PyTorch simple framework to train Optical Character Recognition (OCR) models. 

You can train models to read **captchas**, **license plates**, **digital displays**, and any type of text!

See:

![image](https://user-images.githubusercontent.com/56324869/206953640-087d17b1-a0a7-4f99-ad82-d8c93365bd41.png)


And this framework has rich text while training!

![image](https://user-images.githubusercontent.com/56324869/206952565-1da49dc0-d3ee-4328-8855-19f62aafb435.png)


-----

### Currently Supports:

- CRNNs
- CTC Loss

### Will support :
- Attention at predictions (so you can choose either CTC or Attention)
- Other backbones



## How to train?

----------------------------------------------------

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

- Your data should be of same length, so if your texts have different sizes, use a token of your choice (like '```<pad>```') to pad them to be same length.

- Run:
```sh
python3 train.py
```


# TODO:
- Add logging with hydra, so it saves logging in text files.
- Add CI with github actions, to test if everything works fine after pushes to this repo.
- Add tests to main methods so it keeps secure when adding more models and functionalities in the future.