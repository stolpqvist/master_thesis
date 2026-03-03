from preprocessing.pre_roberta import DataProcessor
from model.roberta import CustomXLMRoberta
from torch.utils.data import DataLoader, Dataset, TensorDataset
import torch
import torch.nn as nn
import numpy as np

class FieldDataset(Dataset):
    def __init__(self, fields, labels):
        self.fields = fields
        self.labels = labels
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
    
        return (
            self.fields[idx],
            self.labels[idx]
        )

class ModelTrain:
    def __init__(self, lr, n_epochs, batch_size, dropout):

        self.lr = lr
        self.epochs = n_epochs
        self.batch_size = batch_size
        self.criterion = nn.CrossEntropyLoss()
        self.dropout = dropout
    
    #How to handle DataProcessor??
    #train_pr = DataProcessor(train_fold)
    #train_pr.label_extractor(label_cl)

    def evaluate(self, model, val_data, label_cl):
        model.eval()
        losses = []
        all_pred = []
        true_labels = []
        
        val_pr = DataProcessor(val_data)
        val_pr.label_extractor(label_cl)

        val_data = val_pr.preprocessing()

        val_tensor = TensorDataset(val_data)

        #with torch.no_grad():
           # for fields, b_labels in da


    def training_loop(self, data, label_cl):

        #extr num_labels
        train_labels = data[label_cl] #NOT tensors
        num_classes = len(np.unique(train_labels))

        #Prepare the data:
        train_pr = DataProcessor(data)
        train_pr.label_extractor(label_cl)


        train_data = train_pr.preprocessing() #Tensors # list of 5 dicts, each (N, 512)

        #Labels to tensors

        label_tensor = torch.tensor(list(train_pr.label2id[label] for label in train_labels), dtype=torch.long)

        #Init the model

        model = CustomXLMRoberta(
                    num_classes=num_classes,
                    hidden_dropout=self.dropout
                )   
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.lr)
        
        model.train()

        
        #Building datasets
        train_dataset = FieldDataset(train_data, label_tensor)
        train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=False)

        total_losses = 0
        all_preds = []
        all_labels = []
        # Main
        # -Trainer
        # -- Stratify

        for epoch in range(self.epochs):
                        
            for fields, b_labels in train_dataloader:

                optimizer.zero_grad()

                logits = model(fields) #predictions
                loss = self.criterion(logits, b_labels)

                loss.backward()
                optimizer.step()

                total_losses += loss.item()
                preds = torch.argmax(logits, dim=1)
                
                all_preds.extend(preds.tolist())
                all_labels.extend(b_labels.tolist())
            
            self.evaluate(model, val_data, label_cl)
            #accuracy per epoch
            correct = sum(pred == label for pred, label in zip(all_preds, all_labels))
            accuracy = correct / len(all_labels)

            print(f"Epoch {epoch + 1}/ {self.epochs}, loss: {total_losses:.4f}, accuracy: {accuracy:.4f}")
        
        #return model

        #EVALUATION




