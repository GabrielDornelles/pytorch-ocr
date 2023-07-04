import torch
from torch import nn
import torch.nn.functional as F


class SelfAttention(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()

        self.query = nn.Linear(input_dim, hidden_dim)
        self.key = nn.Linear(input_dim, hidden_dim)
        self.value = nn.Linear(input_dim, hidden_dim)

    def forward(self, x):
        # Calculate the query, key, and value
        query = self.query(x)
        key = self.key(x)
        value = self.value(x)

        # Calculate the dot product between the query and key
        dot_product = torch.matmul(query, key.transpose(-2, -1))

        # Scale the dot product
        scale = 1.0 / (query.size(-1) ** 0.5)
        dot_product = scale * dot_product

        # Apply the softmax function
        attention_weights = F.softmax(dot_product, dim=-1)

        # Calculate the weighted sum of the value
        weighted_sum = torch.matmul(attention_weights, value)

        return weighted_sum, attention_weights


class Attention(nn.Module):
    """
    https://pytorchnlp.readthedocs.io/en/latest/_modules/torchnlp/nn/attention.html

    Likewise implementation of general attention from torch.nlp. Apparently this is called Bahdanau attention.
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
