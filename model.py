import torch
from torch import nn
from torch.nn import functional as F
import config


class OcrModel(nn.Module):
    def __init__(self, num_chars):
        super().__init__()
        self.conv_1 = nn.Conv2d(3, 256, kernel_size=(3, 5), padding=(1, 1))
        self.pool_1 = nn.MaxPool2d(kernel_size=(2, 2))
        self.conv_2 = nn.Conv2d(256, 64, kernel_size=(3, 5), padding=(1, 1))
        self.pool_2 = nn.MaxPool2d(kernel_size=(2, 2))
        self.linear_1 = nn.Linear(960, 128)
        self.drop_1 = nn.Dropout(0.2)
        self.lstm = nn.GRU(128, 64, bidirectional=True, num_layers=1, dropout=0.25, batch_first=True) 
        self.output = nn.Linear(128, num_chars + 1)

    def forward(self, images, targets=None):
        bs, _, _, _ = images.size()
        x = F.relu(self.conv_1(images))
        x = self.pool_1(x)
        x = F.relu(self.conv_2(x))
        x = self.pool_2(x)
        x = x.permute(0, 3, 1, 2)
        x = x.view(bs, x.size(1), -1)
        #print(f"Shape before Linear layer: {x.shape}")
        x = F.relu(self.linear_1(x))
        x = self.drop_1(x)
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
       


if __name__ == "__main__":
    pass
    #debug shapes
    #cm = OcrModel(33)
    #img = torch.rand((1, 3, 50, 200))
    #x, _ = cm(img, torch.rand((1, 5)))
