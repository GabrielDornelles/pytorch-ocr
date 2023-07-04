import torch


def decode_padded_predictions(predictions: torch.Tensor, classes: list, pad_token: str = "∅") -> list:
    """
    Decode padded predictions into a list of strings.

    Parameters:
        predictions (torch.Tensor): A tensor of predictions with shape (batch_size, sequence_length, num_classes).
        classes (list): A list of classes, where each class is a string.
        pad_token (str, optional): The padding token used in the predictions. Default is '∅'.

    Returns:
        texts (list): A list of strings, where each string is the decoded prediction for a sample in the batch.
    """
    predictions = torch.softmax(predictions, 2)
    predictions = torch.argmax(predictions, 2)
    predictions = predictions.detach().cpu().numpy()

    texts = []
    for item in predictions:
        string = ""
        for idx in item:
            string += classes[idx]
        texts.append(string.replace(pad_token, ""))

    return texts


def decode_predictions(predictions: torch.Tensor, classes: list, blank_token: str = "∅") -> list:
    """
    There's probably faster implementations, I just wrote it myself
    given the below description.

    I use '∅' to denote the blank token.

    ---
    CTC RNN Layer decoder.
    1.
        2 (or more) repeating digits are collapsed into a single instance of that digit unless
        separated by blank - this compensates for the fact that the RNN performs a classification
        for each stripe that represents a part of a digit (thus producing duplicates)
    2.
        Multiple consecutive blanks are collapsed into one blank - this compensates
        for the spacing before, after or between the digits

    Parameters:
        predictions (torch.Tensor): A tensor of predictions with shape (batch_size, sequence_length, num_classes).
        classes (list): A list of classes, where each class is a string.
        blank_token (str, optional): The token used by CTC to represent empty space. Default is '∅'.

    Returns:
        texts (list): A list of strings, where each string is the decoded prediction for a sample in the batch.
    @author Gabriel Dornelles
    """
    predictions = predictions.permute(1, 0, 2)
    predictions = torch.softmax(predictions, 2)
    predictions = torch.argmax(predictions, 2)
    predictions = predictions.detach().cpu().numpy()
    texts = []
    for i in range(predictions.shape[0]):
        string = ""
        batch_e = predictions[i]

        for j in range(len(batch_e)):
            string += classes[batch_e[j]]

        string = string.split(blank_token)
        string = [x for x in string if x != ""]
        string = [list(set(x))[0] for x in string]
        texts.append("".join(string))
    return texts
