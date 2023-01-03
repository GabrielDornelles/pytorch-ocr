import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import resnet18, ResNet18_Weights
from models.attention import Attention
from torch import Tensor

class CRNN(nn.Module):
    """
    CRNN Composed of: 
        - Pretrained ResNet18 until layer 1
        - Bidirectional 1 layer Gated Recurrent Unit (GRU)
        - Attention mechanism at prediction
    
    Parameters:
        dims: number of dimensions the be used at the bottleneck (Linear before GRU, GRU and Attention)
    """
    def __init__(self, dims: int = 256, 
                num_chars: int = 35,
                use_attention: bool = True, 
                use_ctc: bool = True):
        super().__init__()
        self.use_ctc = use_ctc
        self.use_attention = use_attention
        self.num_classes = num_chars + 1

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # feature extraction
        self.convnet = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.linear = nn.Linear(832, dims)
        self.drop = nn.Dropout(0.5)

        # sequence modeling
        self.lstm = nn.GRU(dims, dims//2, bidirectional=True, num_layers=1, batch_first=True) 
        
        # output
        if use_attention:
            self.attention = Attention(dims=dims)
        self.projection = nn.Linear(dims, num_chars + 1) # classes + blank token

        if not use_ctc:
            self.cross_entropy = nn.CrossEntropyLoss().to(self.device)
        

    def encode(self, x):
        # inference through resnet
        x = self.convnet.conv1(x)
        x = self.convnet.bn1(x)
        x = self.convnet.relu(x)
        x = self.convnet.maxpool(x)
        x = self.convnet.layer1(x) # torch.size([bs,64,13,45])
       
        # Permute to stack dimensions and flatten into 2d 
        x = x.permute(0, 3, 1, 2)
        x = x.view(x.size(0), x.size(1), -1) # torch.Size([bs, 45, 832])
     
        features = self.drop(F.relu(self.linear(x))) # [bs,45,256]: 45x256 vector of image features
        # 45 is going to be the sequence length
        return features

    def forward(self, images, targets=None):
        """
        1. Encodes the batch of images into a feature vector of size torch.size([bs,45,256]).
        2. Sequence modeling with Gated Recurrent Unit that processes the feature vector and produce an output at each time step.
        3. (optional) Attention mechanism applied over the time steps produced by GRU.
        4. Project Attention layer (or GRU directly) to linear with classes prob distribuition.
        """
        features = self.encode(images)
        hiddens, _ = self.lstm(features) # [1,45,256]: batch_size, sequence_length, hidden_dim

        if self.use_attention:
            attention , _ = self.attention(hiddens, features) # torch.size([1,45,256]), 45 is the sequence length
            x = hiddens * attention

        x = self.projection(x) if self.use_attention else self.projection(hiddens)
        
        if self.use_ctc and targets is not None:
            x = x.permute(1, 0, 2)
            loss = self.ctc_loss(x,targets)
            return x, loss

        if targets is not None:
            loss = self.nll_loss(x,targets)
            return x, loss
      
        return x, None
    
    
    def pad_targets(self, targets: Tensor, sequence_length: int, one_hot: bool = False) -> Tensor:
        """
        Pad the targets and convert them to one-hot encoding.

        The targets are padded with zeros to reach the specified `sequence_length`.
        Then, they are converted to one-hot encoding using the specified `num_classes`.

        Args:
            targets (torch.Tensor): The targets to pad and convert to one-hot encoding.
                The shape of the tensor should be `(batch_size, len(target))`, where len(target)
                is the length of the string in the image (not padded).
            sequence_length (int): The length to pad the targets to.
            num_classes (int): The number of classes in the one-hot encoding.

        Returns:
            torch.Tensor: The padded and one-hot encoded targets.
                The shape of the returned tensor is (batch_size, sequence_length, num_classes).
        """
        padding = (0, sequence_length - targets.shape[1])
        padded_targets = F.pad(targets, padding, 'constant', 0)
        
        if one_hot:
            one_hot = F.one_hot(padded_targets, self.num_classes)
            return one_hot.to(torch.float)
        return padded_targets
    
    def nll_loss(self, x, targets):
        targets = self.pad_targets(targets, sequence_length=45, one_hot=False)
        scalar = 20
        loss = self.cross_entropy(x.view(-1, x.shape[-1]), targets.contiguous().view(-1)) * scalar
        # I just scaled the loss by a scalar (20) because it starts too low, which difficults optimization.
        # It quickly learn that most of the text is padded, thus making a lot of 'correct' predicitons
        return loss
    
    @staticmethod
    def ctc_loss(x, targets):
        batch_size = x.size(1)
       
        log_probs = F.log_softmax(x, 2)
    
        input_lengths = torch.full(
            size=(batch_size,), fill_value=log_probs.size(0), dtype=torch.int32
        )
    
        target_lengths = torch.full( 
            size=(batch_size,), fill_value=targets.size(1), dtype=torch.int32
        )
      
        loss = nn.CTCLoss(blank=0)(
            log_probs, targets, input_lengths, target_lengths
        )
        return loss




if __name__ == "__main__":
    x = torch.randn((1,3,50,180))

    model = CRNN(dims=256,
        num_chars=35, 
        use_attention=True,
        use_ctc=True
    )

    output = model(x)
