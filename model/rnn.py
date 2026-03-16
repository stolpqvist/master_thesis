import torch
import torch.nn as nn


class RNN(nn.Module):
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
    
    def init_weights(self):

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
        

    def forward(self, input_tensor):
        
        #LSTM with BOTH hidden state and cell state
        batch_size = input_tensor.size(0)
        #print("We are in forward pass")
        #print(f"[DEBUG] input shape: {input_tensor.shape}")  # add temporarily to verify
    
        hidden = torch.zeros(1, batch_size, self.hidden_size, device=input_tensor.device)
        cell = torch.zeros(1, batch_size, self.hidden_size, device=input_tensor.device)

        #process all at once
        emb = self.embedding(input_tensor)
        emb = self.dropout(emb)
        encoder_outputs, (hidden, cell) = self.rnn(emb, (hidden, cell))

        #emb.permute(1, 0, 2)
       
        #encoder_outputs = encoder_outputs.permute(1, 0, 2)

        #all_hiddens = []
        #print(input_tensor.size(1))
        #for i in range(input_tensor.size(1)): #iterate over seq_len, instead of batch
            #emb = self.embedding(input_tensor[i]).view(1, 1, -1)
            #emb = self.embedding(input_tensor[:, i]).unsqueeze(1)
            #emb = self.dropout(emb)
            #_, (hidden, cell) = self.rnn(emb, (hidden, cell))
            #all_hiddens.append(hidden)

        #print(all_hiddens)

        #encoder_outputs = torch.cat(all_hiddens, dim=1) # (batch, seq_len, hidden_size)

        #use final hidden state as query
        #query = hidden.transpose(0, 1) # (batch, 1, hidden_size)
        query = hidden
        #attention and classifying
        context = self.attn(encoder_outputs, query)
        #context = self.attn(context)
        context = context.squeeze(1)
        logits = self.classifier(context)
        #print("We are in RNN")
        return logits
    
class AttentionConcat(nn.Module):

    def __init__(self, hidden_size):
        super(AttentionConcat, self).__init__()

        self.attn = nn.Linear(hidden_size * 2, hidden_size)
        self.vector = nn.Parameter(torch.rand(hidden_size))
    
    def forward(self, output_enc, hidden_dec):
        #output_enc: (1, seq_len, hidden_size) - all hidden states
        #Hidden_dec: (1, 1, hidden_size) - the query (final hidden state)
        #hidden_dec = hidden_dec.squeeze(0)

        hidden_dec = hidden_dec.transpose(0, 1)
        len_source = output_enc.shape[1]
        hidden_dec = hidden_dec.repeat(1, len_source, 1)

        concated_tensor = torch.cat((output_enc, hidden_dec), dim=2)
        attention_energies = torch.tanh(self.attn(concated_tensor))
        scores = torch.matmul(attention_energies, self.vector).unsqueeze(1)
        attn_weights = nn.Softmax(dim=-1)(scores)
        ctx_vec = torch.bmm(attn_weights, output_enc)

        return ctx_vec #(1, 1, hidden_size)