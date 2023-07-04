import pytest
import torch
from utils.model_decoders import decode_predictions, decode_padded_predictions


@pytest.mark.parametrize("expected_text", ["012345", "543210", "332253", "666222"])
@pytest.mark.parametrize("sequence_length", [45, 100, 1000])
def test_decode_predictions(expected_text, sequence_length):
    classes = ["∅", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
    batch_size = 1
    num_classes = len(classes)
    predictions = torch.zeros(sequence_length, batch_size, num_classes)
    # Calculate the number of ∅ characters to insert
    num_blanks = sequence_length - len(expected_text)

    # Calculate the interval between ∅ characters
    interval = num_blanks // (len(expected_text) - 1) if len(expected_text) > 1 else num_blanks

    # Set the values in the predictions tensor based on the expected text
    for i, char in enumerate(expected_text):
        class_index = classes.index(char)
        predictions[i * interval, 0, class_index] = 1.0

    decoded_texts = decode_predictions(predictions, classes)

    assert decoded_texts == [expected_text]


# def test_decode_padded_predictions(expected_text, sequence_length):
#     pass
