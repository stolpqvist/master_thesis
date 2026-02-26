from preprocessing.pre_roberta import DataProcessor
from model.roberta import CustomXLMRoberta
import torch
import torch.nn as nn

class ModelTrain:
    def __init__(self, data, labels, learning_rate, n_epochs):
        self.dataset = data
        self.labels = labels
        self.lr = learning_rate
        self.epochs = n_epochs
        self.model = CustomXLMRoberta(num_classes=len(self.labels))
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr)
    

    def training_loop(self, model):

        model.train()

        total_losses = 0
        all_preds = []
        all_labels = []

        for epoch in range(self.epochs()):
            
            for data, labels in self.dataset:
                self.optimizer.zero_grad()

                logits = self.model(data) #batch, num_classes
                loss = self.criterion(logits, labels)

                loss.backward()
                self.optimizer.step()

                total_losses += loss.item()
                preds = torch.argmax(logits, dim=1)
                all_preds.extend(preds.tolist())
                all_labels.extend(labels.tolist())
            
            #accuracy per epoch
            correct = sum(pred == label for pred, label in zip(all_preds, all_labels))
            accuracy = correct / len(all_labels)

            print(f"Epoch {epoch + 1}/ {self.epochs} - loss: {total_losses:.4f}, accuracy: {accuracy:.4f}")




