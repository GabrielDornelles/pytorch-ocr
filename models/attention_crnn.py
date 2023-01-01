import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import resnet18, ResNet18_Weights


class Attention(nn.Module):
    """
    https://pytorchnlp.readthedocs.io/en/latest/_modules/torchnlp/nn/attention.html

    Likewise implementation of general attention from torch.nlp
    """
    def __init__(self, dims):
        super().__init__()
        self.linear_in = nn.Linear(in_features=dims, out_features=dims, bias=False)
        self.linear_out = nn.Linear(in_features=dims * 2, out_features=dims, bias=False)
        self.softmax = nn.Softmax(dim=-1)
        self.tanh = nn.Tanh()
    
    def forward(self, query, context):
        """
        Args:
            query (:class:`torch.FloatTensor` [batch size, output length, dimensions]): Sequence of
                queries to query the context.
            context (:class:`torch.FloatTensor` [batch size, query length, dimensions]): Data
                overwhich to apply the attention mechanism. In the OCR case, context is the image features 
                before lstm.

        Returns:
            :class:`tuple` with `output` and `weights`:
            * `output` (:class:`torch.LongTensor` [batch size, output length, dimensions]):
              Tensor containing the attended features.
            * `weights` (:class:`torch.FloatTensor` [batch size, output length, query length]):
              Tensor containing attention weights.
        """
        batch_size, output_len, dims = query.size()
        query_len = context.size(1)

        query = query.reshape(batch_size * output_len, dims)
        query = self.linear_in(query)
        query = query.reshape(batch_size, output_len, dims)

        attention_scores = torch.bmm(query, context.transpose(1, 2).contiguous())
        
        attention_scores = attention_scores.view(batch_size * output_len, query_len)
        attention_weights = self.softmax(attention_scores)
        attention_weights = attention_weights.view(batch_size, output_len, query_len)

        mix = torch.bmm(attention_weights, context)
        combined = torch.cat((mix, query), dim=2)
        combined = combined.view(batch_size * output_len, 2 * dims)

        output = self.linear_out(combined).view(batch_size, output_len, dims)
        output = self.tanh(output)
        return output, attention_weights



class ResNetGruAttention(nn.Module):
    """
    CRNN Composed of: 
        - Pretrained ResNet18 until layer 1
        - Bidirectional 1 layer Gated Recurrent Unit (GRU)
        - Attention mechanism at prediction
    
    """
    def __init__(self, dims: int = 256, num_chars: int = 35):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # feature extraction
        self.convnet = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.linear = nn.Linear(832, dims)
        self.drop = nn.Dropout(0.5)

        # sequence modeling
        self.lstm = nn.GRU(dims, dims//2, bidirectional=True, num_layers=1, batch_first=True) 
        
        # output
        self.attention = Attention(dims=dims)
        self.projection = nn.Linear(dims, num_chars + 1) # classes + blank token
        
       
    def _char_to_onehot(self, input_char, onehot_dim=45):
        #input_char = input_char.unsqueeze(1)
        batch_size = input_char.size(0)
        one_hot = torch.LongTensor(batch_size, onehot_dim).zero_().to(self.device)
        one_hot = one_hot.scatter_(1, input_char, 1)
        return one_hot

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
        features = self.drop(F.relu(self.linear(x)))
        x, _ = self.lstm(features)
        x, _ = self.attention(x, features)
        x = self.projection(x) # project attention back to linear layer with classes prob distribuition
        x = x.permute(1, 0, 2) # should be torch.Size([45, 8, 36])
        if targets is not None:
            loss = self.ctc_loss(x,bs,targets)
            return x, loss
            
       
        return x, None
    
    def nll_loss(x, targets):
        # TODO: implement attention + NLL Loss
        # log_probs = F.log_softmax(x, 2)
        # log_probs = log_probs.view(log_probs.shape[0] * log_probs.shape[1], -1)

        # targets = self._char_to_onehot(targets)
        # targets = targets.view(-1)

        # loss = self.attn_loss(log_probs, targets)#, input_lengths, target_lengths)
        #print(loss)
        loss = None
        return x, loss
    
 
    def ctc_loss(self, x, batch_size, targets):
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

    model = ResNetGruAttention(dims=256,num_chars=35)

    output = model(x)
