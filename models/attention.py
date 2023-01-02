import torch
from torch import nn

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
        # query=Q, context=K
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