import torch
import torch.nn as nn

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(RNN, self).__init__()

        #Encoder part
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.rnn = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.attn = AttentionConcat(hidden_size) #importing this 
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, input_tensor):
        
        #LSTM with BOTH hidden state and cell state
        batch_size = input_tensor.size(0)

        #print(f"[DEBUG] input shape: {input_tensor.shape}")  # add temporarily to verify
    
        hidden = torch.zeros(1, batch_size, self.hidden_size, device=input_tensor.device)
        cell = torch.zeros(1, batch_size, self.hidden_size, device=input_tensor.device)

        emb = self.embedding(input_tensor)
        #emb.permute(1, 0, 2)
        encoder_outputs, (hidden, cell) = self.rnn(emb, (hidden, cell))
        #encoder_outputs = encoder_outputs.permute(1, 0, 2)

        #all_hiddens = []

        #for i in range(input_tensor.size(1)): #iterate over seq_len, instead of batch
        #    emb = self.embedding(input_tensor[:, i]).unsqueeze(0)
        #    _, (hidden, cell) = self.rnn(emb, (hidden, cell))
        #    all_hiddens.append(hidden)
        

        #encoder_outputs = torch.cat(all_hiddens, dim=0).permute(1, 0, 2) # (batch, seq_len, hidden_size)

        #use final hidden state as query
        query = hidden.transpose(0, 1) # (batch, 1, hidden_size)

        #attention and classifying
        context = self.attn(encoder_outputs, query)
        logits = self.classifier(context.squeeze(1))
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
        len_source = output_enc.shape[1]
        hidden_dec = hidden_dec.repeat(1, len_source, 1)
        concated_tensor = torch.cat((output_enc, hidden_dec), dim=2)
        attention_energies = torch.tanh(self.attn(concated_tensor))
        scores = torch.matmul(attention_energies, self.vector).unsqueeze(1)
        attn_weights = nn.Softmax(dim=-1)(scores)
        ctx_vec = torch.bmm(attn_weights, output_enc)

        return ctx_vec #(1, 1, hidden_size)