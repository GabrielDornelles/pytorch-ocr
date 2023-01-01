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