# TorchNN-OCR

*Rich Text while training*

![image](https://user-images.githubusercontent.com/56324869/146463494-b3a2ef46-cb07-44e3-b6c4-02d8841927df.png)


## A Pytorch Convolutional Recurrent Neural Network with CTC Loss.

----------------------------------------------------

- Just create a directory called "dataset" and throw your images there (preferable to be png, but you can use other formats as long as you change [that](https://github.com/GabrielDornelles/EchidNet-OCR/blob/5275b1169051763fbb08f583871a28e88c706454/train.py#L56))

- Make sure to normalize your data length (num of chars in image), or just trick it to send same length adding any character to represent empty space.

- Run:
```sh
python3 train.py
```
----------------------------------------------------
**NOTE**: Your images must be named like "text_in_your_img.png", model will pick up the text on it as the answer and train based on that.

It will save accuracy and losses graphs in root directory

![accuracy_graph](https://user-images.githubusercontent.com/56324869/111052708-1af37880-843c-11eb-9b27-954519e5975e.png)

![losses_graph](https://user-images.githubusercontent.com/56324869/111052721-39597400-843c-11eb-8d22-d26f815d4209.png)
