"""This is the module for the CNN model."""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import sys
class ClassificationCNN(nn.Module):
    """
    This class deals with the setup and training of the CNN model.

    Attributes:
        cl_1 (nn.Conv1d) = The first convolutional layer.
        cl_2 (nn.Conv1d) = The second convolutional layer.
        cl_3 (nn.Conv1d) = The third convolutional layer.
        cl_4 (nn.Conv1d) = The fourth convolutional layer.
        input_projection (nn.Linear) = A intermetiate fully connected layer.
        embedding (nn.Embedding) = The embedding layer.
        inter_fc (nn.Linear) =  The last fully connected layer.
        classificer (nn.Linear) = The actual classifier.
        dropout (nn.Dropout) = The dropout layer.

    Methods:
        forward(application:torch.tensor)
    """
    def __init__(self, input_size, num_classes, embedding_dim = 300, hidden_size = 512, dropout = 0.04):
        super().__init__()
        #Initialises convolutional layers with various amount of context.
        self.cl_1 = nn.Conv1d(hidden_size, hidden_size//2, kernel_size = 3, padding = 1)
        self.cl_2 = nn.Conv1d(hidden_size, hidden_size//2, kernel_size = 5, padding = 2)
        self.cl_3 = nn.Conv1d(hidden_size, hidden_size//2, kernel_size = 7, padding = 3)
        self.cl_4 = nn.Conv1d(hidden_size, hidden_size//2, kernel_size = 9, padding = 4)
 
        #Initialises intermediate fully-connected layer.
        self.input_projection = nn.Linear(embedding_dim, hidden_size)

        self.embedding = nn.Embedding(input_size, embedding_dim, padding_idx=0) 
        #Initialises final fully connected layer.
        self.inter_fc = nn.Linear(hidden_size * 2, hidden_size)
        self.classifier = nn.Linear(hidden_size, num_classes)
        #Initialises dropout layer.
        self.dropout = nn.Dropout(dropout)

        #Initialises activation function.
        self.relu = nn.ReLU()

    def forward(self, application:torch.tensor) -> torch.Tensor:
        """
        This method deals with forward pass for the CNN model.

        Arguments:
            application (torch.tensor) = The application to be processed.

        Returns:
            prediction
        """
        emb = self.dropout(self.embedding(application)) 
        projected = self.relu(self.input_projection(emb))
        features = projected.permute(0, 2, 1)

        c1 = F.max_pool1d(self.relu(self.cl_1(features)), features.size(2)).squeeze(2)
        c2 = F.max_pool1d(self.relu(self.cl_2(features)), features.size(2)).squeeze(2)
        c3 = F.max_pool1d(self.relu(self.cl_3(features)), features.size(2)).squeeze(2)
        c4 = F.max_pool1d(self.relu(self.cl_4(features)), features.size(2)).squeeze(2)
        #Applies convolutional layers in combination with relu activation.

        #Concatenates the outputs from the additional layers.
        pooled = torch.cat((c1, c2, c3, c4), dim=1)
        
        
        #Apply the intermediate connected layer and dropout.
        output = self.dropout(self.relu(self.inter_fc(pooled)))
        #Apply the fully connected layer.
        predictions = self.classifier(output)

        return predictions


