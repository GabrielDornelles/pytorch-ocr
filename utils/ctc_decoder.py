import torch


def decode_predictions(predictions: torch.Tensor, classes: list, blank_token: str = '∅') -> list:
    """
    @author Gabriel Dornelles
    
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

    params:
        inputs: torch.tensor batch of predictions
        classes: list of classes (letters, numbers etc) to predict
    
    returns:
        list of decoded predictions as strings
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
        texts.append(''.join(string))
    return texts