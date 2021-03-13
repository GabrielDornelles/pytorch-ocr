# EchidNet-OCR
## A Torch Optical Character Recognition Neural Net using EchidNet,  and other CNN's with a Recurrent layer on top of it.


----------------------------------------------------
1 - Just create a dir called "dataset" and throw your images there (preferable to be png, but you can use other formats as long as you change

2 - Make sure to normalize your data length (num of chars in image) while training (or just trick it to send same length). You can also use the image_validation.py and specify how big the length can be and just remove data out of expected.

3 - After that just run train.py and you should be happy.


# TODO: 
If you want to do any of these: [EchidNet Model](https://github.com/GabrielDornelles/EchidNet)
- EchidNet for simple documents and text  not implemented yet, instead working with a different CNN for more complicated characters with noise. 
- Contour finding to use EchidNet at max speed without the Recurrent layer (for simple letters and numbers  that can be detected and splitted)
- Hyperparameter settings for EchidNet or model variations with stronger feature extraction (but still small).
- Larger training for both simple and noisy characters.
