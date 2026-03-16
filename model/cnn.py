
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import sys
from tqdm import tqdm
import argparse
import subprocess
class ClassificationCNN(nn.Module):

    def __init__(self, vocab_size, num_classes, embedding_dim = 300, hidden_dim = 512, dropout = 0.04):
        super().__init__()
        #Initialises convolutional layers with various amount of context.
        self.cl_1 = nn.Conv1d(hidden_dim, hidden_dim//2, kernel_size = 3, padding = 1)
        self.cl_2 = nn.Conv1d(hidden_dim, hidden_dim//2, kernel_size = 5, padding = 2)
        self.cl_3 = nn.Conv1d(hidden_dim, hidden_dim//2, kernel_size = 7, padding = 3)
        self.cl_4 = nn.Conv1d(hidden_dim, hidden_dim//2, kernel_size = 9, padding = 4)
 
        #Initialises intermediate fully-connected layer.
        self.inter_projection = nn.Linear(embedding_dim, hidden_dim)

        self.embedding = nn.Embedding(vocab_size, embedding_size, padding_idx=0) 
        #Initialises final fully connected layer.
        self.inter_fc = nn.Linear(hidden_dim * 2, hidden_dim)
        self.classifier = nn.Linnear(hidden_dim, num_classes)
        #Initialises dropout layer.
        self.dropout = nn.Dropout(dropout)

        #Initialises activation function.
        self.relu = nn.ReLU()

    def forward(self, application:torch.tensor):
        emb = self.dropout(self.embedding(application)) 
        projected = self.relu(self.input_projection(emb))
        features = projected.permute(0, 2, 1)

        c1 = F.maxpool1d(self.relu(self.cl_1(features)), features.size(2)).squeeze(2)
        c2 = F.maxpool1d(self.relu(self.cl_2(features)), features.size(2)).squeeze(2)
        c3 = F.maxpool1d(self.relu(self.cl_3(features)), features.size(2)).squeeze(2)
        c4 = F.maxpool1d(self.relu(self.cl_4(features)), features.size(2)).squeeze(2)
        #Applies convolutional layers in combination with relu activation.

        #Concatenates the outputs from the additional layers.
        pooled = torch.cat((c1, c2, c3, c4), dim=1)
        
        
        #Apply the intermediate connected layer and dropout.
        output = self.dropout(self.relu(self.inter_fc(pooled)))
        #Apply the fully connected layer.
        predictions = self.classifier(output)

        return predictions


