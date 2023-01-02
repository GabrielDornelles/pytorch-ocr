# Changelog 

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

## [1.0.1] - 01/01/2023

### Added
- Added Attention mechanism at prediction stage of CRNN (available at models/attention_crnn.py)

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
