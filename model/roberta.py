import torch
from transformers import AutoModel, AutoConfig
import torch.nn as nn
from preprocessing.pre_roberta import DataProcessor



class CustomXLMRoberta(nn.Module):
    def __init__(self, model_name="xlm-roberta-base", hidden_dropout=0.1):
        super().__init__()

        self.config = AutoConfig.from_pretrained(model_name)
        self.backbone = AutoModel.from_pretrained(model_name)

        #Custom head

        self.projection = nn.Linear(self.config.hidden_size, 256)
        self.dropout = nn.Dropout(hidden_dropout)
    

    def forward(self, input_ids, attention_mask = None, token_type_ids = None):
        outputs = self.backbone(
            input_ids = input_ids,
            attention_mask = attention_mask,
            token_type_ids = token_type_ids,
            return_dict = True
        )
        
        cls_embedding = outputs.last_hidden_state[:,0,:]

        x = self.dropout(cls_embedding)
        x = self.projection(x)
        return x
    
    def pooling(self, hidden_mask):
        mask = mask.unsqueeze(-1).float() # (batch, seq, 1)
        summed = (hidden * mask).sum(dim=1)
        counts = mask.sum(dim=1)
        return summed/counts




class Robsterbobster():
    def __init__(self, labels):
        self.labels = labels
        
#What we could do
#Build a custom class for the roberta
#In the class:
    # Run through the process of setting it up
    # Fix the vocabulary, unless it is shared amongst other classes
    # At which point we would create a separate class for it

