# TorchNN-OCR
## A Torch Optical Character Recognition Neural Net using EchidNet,  and other CNN's with a Recurrent layer on top of it.


----------------------------------------------------
1 - Just create a dir called "dataset" and throw your images there (preferable to be png, but you can use other formats as long as you change [that](https://github.com/GabrielDornelles/EchidNet-OCR/blob/5275b1169051763fbb08f583871a28e88c706454/train.py#L56))

2 - Make sure to normalize your data length (num of chars in image) while training (or just trick it to send same length). You can also use the image_validation.py and specify how big the length can be and just remove data out of expected.

3 - After that just run train.py and you should be happy.

NOTE: Your images must be named like "text_in_your_img.png", model will pick up the text on it as the answer and train based on that.

You should have your model saved as the desired name in config file and two png files containing accuracy over epochs and train_loss&test_loss over epochs. Change the config file and do your custom training (there no calculation for the convolutional layers, if you change image size make sure to also change the number of parameters in the linear layer before Gated Recurrent Unit ([here](https://github.com/GabrielDornelles/EchidNet-OCR/blob/11d07be575898eeae8d731fab95183f91a005019/model.py#L43))). Also you can Use LSTM instead of GRU or just create new models.

your model data will look like this (example):

![accuracy_graph](https://user-images.githubusercontent.com/56324869/111052708-1af37880-843c-11eb-9b27-954519e5975e.png)

![losses_graph](https://user-images.githubusercontent.com/56324869/111052721-39597400-843c-11eb-8d22-d26f815d4209.png)

demo:
https://user-images.githubusercontent.com/56324869/111344173-e9a4c380-865a-11eb-99db-4d990fe2dd01.mp4


# TODO: 
If you want to do any of these here's the notebook: [EchidNet Model](https://github.com/GabrielDornelles/EchidNet)
- EchidNet for simple documents and text  not implemented yet, instead working with a different CNN for more complicated characters with noise. 
- Contour finding to use EchidNet at max speed without the Recurrent layer (for simple letters and numbers  that can be detected and splitted)
- Hyperparameter settings for EchidNet or model variations with stronger feature extraction (but still small).
- Larger training for both simple and noisy characters.
