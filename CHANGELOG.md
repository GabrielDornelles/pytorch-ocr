# Changelog 

## [1.0.3] - 03/01/2023
### Highlight

- Training a model with attention and cross entropy loss is now finally possible.

<p align="center">
Trained with Cross Entropy Loss and Attention:
</p>

<p align="center">
  <img src="https://user-images.githubusercontent.com/56324869/210393213-99d35d57-4c85-41a5-b98f-df08c76955fe.png" />
</p>


### Bugfixes
- Attention layer is now properly applied to the hidden states of GRU Unit. Before, attention was directly projected to linear layer, now Gated Recurrent Unit hidden states are multiplied by the attention weights and then projected to linear layer.
- CrossEntropyLoss is now properly working. Before it was calculating probabilistic wise the loss between two 3d tensors, now tensors are reshaped to work like in image classification, i.e the target is now reshaped to a 1d **long** tensor (not float) that holds the batch of targets in sequence, and only the correct indexes are there, not one hot enconding like before (target is tensor of shape: batch_size * sequence_length).
- Cross Entropy Loss was tested and achieved 80% accuracy on same dataset with same model as using CTC Loss. Cross Entropy Loss has to be multiplied by some scalar to compensate the fact that padding is quickly learned by the model. In the case I tested, targets had 6 characters and 39 pad tokens, thus, the loss is very low at the very start of the training, because 39 classifications are always correct and learned very fast since thats a strong bias.

### Added
- Added gradient clipping in the training procedure (torch.nn.utils.clip_grad_norm_) with default of 5. It's not parameterizable as I didn't see the necessity of it, but it's at ```engine.py.train_fn``` right after ```loss.backward()```

## [1.0.2] - 02/01/2023

### Changed
- Attention network is now at models/attention.py.
- Refactored ugly variable names in train.py.

### Added
- Cross Entropy loss is now available to use with Attention. Attention is also optional. To use it simply create the model with:
    ```py
    from models.crnn import CRNN

    model = CRNN(dims=256,
        num_chars=35, 
        use_attention=True,
        use_ctc=True
    )
    ```
- Pad + One Hot Encoding method (used if you want to train with cross entropy loss).
- A decoder (utils/model_decoders.py.decode_padded_predictions) if cross entropy is used (simple output decoder replacing pad token with empty string).
- General documentation for methods and model creation.

### Removed
- Removed the old models as attention and loss functions are now parameterizable.


## [1.0.1] - 01/01/2023

### Added
- Added Attention mechanism at prediction stage of CRNN (available at models/attention_crnn.py)


## [1.0.0] - 12/12/2022

### Changed
- Refactored the whole codebase.

### Added
- Added a new CRNN with ResNet backbone.
- Added hydra configs (before it had python config.py file).
- Added hydra logging which now outputs the training log into a file. Rich tables are displayed while training, but training log like losses and accuracies with timestamps are available at train.log (that is written inside /output/date/time/train.log)
- Added a CTC Decoder method.

### Removed

- config.py was replaced by hydra.
- Old methods to remove duplicates (now using ctc decoder).



