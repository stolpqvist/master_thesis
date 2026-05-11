"""
This is the module for the RNN model and its attention layer.
"""
import torch
import torch.nn as nn


class RNN(nn.Module):
    """
    This class deals initialising the RNN model.

    Attributes:
        hidden_size (int) = The size of the hidden layer.
        embedding (nn.Embedding) = The embedding layer.
        dropout (nn.Dropout) = The dropout layer.
        rnn (nn.LSTM) = The long short-term memory RNN model.
        attn (nn.Module) = The attention layer.
        classifier (nn.Linear) = The classifier layer.
    """
    def __init__(self, input_size, hidden_size, num_classes, dropout):
        super(RNN, self).__init__()

        #Encoder part
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.rnn = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.attn = AttentionConcat(hidden_size) #importing this 
        self.classifier = nn.Linear(hidden_size, num_classes)

        self.init_weights()
    
    def init_weights(self) -> None:
        """
        Initialises the weights for the RNN.
        """
        #Embedding
        nn.init.uniform_(self.embedding.weight, -0.1, 0.1)

        #RNN parameters
        for name, param in self.rnn.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param) #input-hidden weights
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param) #hidden-hidden weights
            elif 'bias' in name:
                nn.init.zeros_(param)
        
        #attention linear layer
        nn.init.xavier_uniform_(self.attn.attn.weight)
        nn.init.zeros_(self.attn.attn.bias)

        #attention scoring vector
        nn.init.uniform_(self.attn.vector, -0.1, 0.1)

        #classifier
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)
        

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        
        #LSTM with BOTH hidden state and cell state
        batch_size = input_tensor.size(0)
    
        hidden = torch.zeros(1, batch_size, self.hidden_size, device=input_tensor.device)
        cell = torch.zeros(1, batch_size, self.hidden_size, device=input_tensor.device)

        #process all at once
        emb = self.embedding(input_tensor)
        emb = self.dropout(emb)
        encoder_outputs, (hidden, cell) = self.rnn(emb, (hidden, cell))

        query = hidden
        #attention and classifying
        context = self.attn(encoder_outputs, query)
        context = context.squeeze(1)
        logits = self.classifier(context)
        #print(f"This is the type of logits: {type(logits)}")
        return logits
    
class AttentionConcat(nn.Module):
    """
    This class deals specifically with the attention layer for the RNN model.

    Attributes:
        attn (nn.Linear) = A linear layer.
        vector (nn.Parameter) = A vector parameter layer.

    Methods:
        forward(output_enc, hidden_dec) -> torch.Tensor
            Deals with the forward pass of the attention layer.
    """
    def __init__(self, hidden_size):
        super(AttentionConcat, self).__init__()

        self.attn = nn.Linear(hidden_size * 2, hidden_size)
        self.vector = nn.Parameter(torch.rand(hidden_size))
    
    def forward(self, output_enc: torch.Tensor, hidden_dec: torch.Tensor) -> torch.Tensor:
        """
        This method deals with the forward pass of the attention layer.

        Arguments:
            output_enc (torch.Tensor) = Full hidden state.
            hidden_dec (torch.Tensor) = Current hidden state at time t.

        Returns:
            ctx_vec (torch.Tensor) = The context vector.
        """
        hidden_dec = hidden_dec.transpose(0, 1)
        len_source = output_enc.shape[1]
        hidden_dec = hidden_dec.repeat(1, len_source, 1)

        concated_tensor = torch.cat((output_enc, hidden_dec), dim=2)
        attention_energies = torch.tanh(self.attn(concated_tensor))
        scores = torch.matmul(attention_energies, self.vector).unsqueeze(1)
        attn_weights = nn.Softmax(dim=-1)(scores)
        ctx_vec = torch.bmm(attn_weights, output_enc)
        return ctx_vec #(1, 1, hidden_size)
