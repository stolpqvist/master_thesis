"""
This module deals with the implementation and initialisation of the xlm-roberta-base model.
"""
import torch
from transformers import AutoModel, AutoConfig
import torch.nn as nn
from preprocessing.pre_roberta import DataProcessor



class CustomXLMRoberta(nn.Module):
    """
    This class handles the initialisation and implementation of the xlm-roberta-base model.

    Attributes:
        config (AutoConfig.Config) = The configuration for the xlm-roberta-base model
        backbone (AutoModel.model) = The backbone of the model.
        pooling_strategy (str) = Which pooling strategy to implement.
        hidden (int) = The size of the hidden layer.
        dropout (nn.Dropout) = The dropout layer.
        projection (nn.Linear) = The projection layer.
        classifier (nn.Linear) = The classifier layer.

    Methods:
        pool(hidden_states, attention_mask) ->

        forward(fields) ->
            Deals with the forward pass and classification for the model.

    """
    def __init__(self, num_classes: int, model_name: str="xlm-roberta-base", hidden_dropout: float =0.1, pooling: str="cls"):
        super().__init__()

        self.config = AutoConfig.from_pretrained(model_name)
        self.backbone = AutoModel.from_pretrained(model_name)
        self.pooling_strategy = pooling
        hidden = self.config.hidden_size
        self.dropout = nn.Dropout(hidden_dropout)
        self.projection = nn.Linear(hidden, 256)
        #self.field_attention = nn.Linear(256, 1)
        self.classifier = nn.Linear(256, num_classes)
    
    
    def pool(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        This method utilises the pooling strategy requested and returns
        pooled contents.

        Arguments:
            hidden_states () =
            attention_mask () =

        Returns:
            hidden_states[:,0,:] () = If conditions are met.
            summed/counts () =
        """
        if self.pooling_strategy == "cls":
            return hidden_states[:, 0, :]
        else: #mean pooling
            mask = attention_mask.unsqueeze(-1).float() # (batch, seq, 1)
            summed = (hidden_states * mask).sum(dim=1)
            counts = mask.sum(dim=1).clamp(min=1e-9) #avoid division by zero
            return summed/counts

    def forward(self, fields:dict) -> torch.Tensor:
        """
        This method deals with the forward pass of the model for classification purposes.

        Arguments:
            fields (torch.Tensor) = The fields selected for this application.

        Returns:
            torch.Tensor = The output of the classification process.
        """
        outputs = self.backbone(
            input_ids = fields["input_ids"],
            attention_mask= fields["attention_mask"],
            return_dict=True
        )

        pooled = self.pool(outputs.last_hidden_state, fields["attention_mask"])
        projected= self.projection(self.dropout(pooled))
        return self.classifier(self.dropout(projected))
  
