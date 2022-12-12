import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import resnet18, ResNet18_Weights


class ResNetGRU(nn.Module):
    """
    CRNN Composed of: 
        - Pretrained ResNet18 until layer 1
        - Bidirectional 1 layer Gated Recurrent Unit (GRU)
    
    """
    def __init__(self, num_chars):
        super().__init__()
        self.convnet = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.linear = nn.Linear(832, 128)
        self.drop = nn.Dropout(0.5)
        self.lstm = nn.GRU(128, 64, bidirectional=True, num_layers=1, batch_first=True) 
        self.output = nn.Linear(128, num_chars + 1) # classes + blank token

    def forward(self, images, targets=None):
        bs, _, _, _ = images.size()
        # inference through resnet
        x = self.convnet.conv1(images)
        x = self.convnet.bn1(x)
        x = self.convnet.relu(x)
        x = self.convnet.maxpool(x)
        x = self.convnet.layer1(x)
        
        # Permute
        x = x.permute(0, 3, 1, 2)
        x = x.view(bs, x.size(1), -1)
        
        # To linear and then to GRU
        x = self.drop(F.relu(self.linear(x)))
        x, _ = self.lstm(x)
        x = self.output(x)
        x = x.permute(1, 0, 2)

        if targets is not None: 
            log_probs = F.log_softmax(x, 2)
        
            input_lengths = torch.full(
                size=(bs,), fill_value=log_probs.size(0), dtype=torch.int32
            )
        
            target_lengths = torch.full( 
                size=(bs,), fill_value=targets.size(1), dtype=torch.int32
            )
            loss = nn.CTCLoss(blank=0)(
                log_probs, targets, input_lengths, target_lengths
            )
            return x, loss

        return x, None
