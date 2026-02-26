import torch
from transformers import AutoModel, AutoConfig
import torch.nn as nn
from preprocessing.pre_roberta import DataProcessor



class CustomXLMRoberta(nn.Module):
    def __init__(self, num_classes, model_name="xlm-roberta-base", hidden_dropout=0.1, pooling="cls"):
        super().__init__()

        self.config = AutoConfig.from_pretrained(model_name)
        self.backbone = AutoModel.from_pretrained(model_name)
        self.pooling_strategy = pooling

        hidden = self.config.hidden_size

        #Custom head

        self.dropout = nn.Dropout(hidden_dropout)
        self.projection = nn.Linear(hidden, 256)
        self.field_attention = nn.Linear(256, 1)
        self.classifier = nn.Linear(256, num_classes)
    
    
    def pool(self, hidden_states, attention_mask):
        if self.pooling_strategy == "cls":
            return hidden_states[:, 0, :] #(batch*5, 768)
        else: #mean pooling
            mask = attention_mask.unsqueeze(-1).float() # (batch, seq, 1)
            summed = (hidden_states * mask).sum(dim=1)
            counts = mask.sum(dim=1).clamp(min=1e-9) #avoid division by zero
            return summed/counts

    def forward(self, fields):
        #fields: list of 5 dicts, each with input_ids and attention_mask of shape (batch, 512)

        #stacking all fields into one big batch: (batch *5, 512)
        all_input_ids = torch.cat([f["input_ids"] for f in fields], dim=0)
        all_masks = torch.cat([f["attention_mask"] for f in fields], dim=0)
        
        
        #One forward pass through xlm-r
        outputs = self.backbone(
            input_ids = all_input_ids,
            attention_mask = all_masks,
            return_dict = True
        )
        
        #pool into one vector per (sample, field) -> (batch*5, 768)
        pooled = self.pool(outputs.last_hidden_state, all_masks)

        #project dowm (batch*5, 256)
        pooled = self.projection(self.dropout(pooled))

        #split back into fields and stack -> (batch, 5, 256)
        cls_per_field = pooled.chunk(len(fields), dim=0)
        stacked = torch.stack(cls_per_field, dim=1)

        #field attention -> weighted sum -> (batch, 256)
        scores = self.field_attention(stacked) #(batch, 5, 1)
        weights = torch.softmax(scores, dim=1)
        combined = (weights * stacked).sum(dim=1) #batch, 256)

        #classify
        return self.classifier(self.dropout(combined)) #(batch, num_classes)


        
#What we could do
#Build a custom class for the roberta
#In the class:
    # Run through the process of setting it up
    # Fix the vocabulary, unless it is shared amongst other classes
    # At which point we would create a separate class for it

