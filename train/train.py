from preprocessing.pre_roberta import DataProcessor
from model.roberta import CustomXLMRoberta
from torch.utils.data import DataLoader, TensorDataset
import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report



class ModelTrain:
    def __init__(self, lr, n_epochs, batch_size, dropout):

        self.lr = lr
        self.epochs = n_epochs
        self.batch_size = batch_size
        self.criterion = nn.CrossEntropyLoss()
        self.dropout = dropout
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")



    def training_loop(self, data, val_data, label_cl):
        #TODO Implement random parameter generation
        #TODO Implement writing to file
        #TODO Implement constructor that constructs folders as needed
        print(f"Training on: {self.device}")

        #extr num_labels
        #train_labels = data[label_cl] #NOT tensors
        #num_classes = len(np.unique(train_labels))

        #Prepare the data:
        train_fold = DataProcessor(data)
        train_fold.label_extractor()

        train_dataloader = DataLoader(
            train_fold,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=False
        )

        print("Data loaded successfully")

        #Init the model

        model = CustomXLMRoberta(
                    num_classes=len(train_fold.label2id.keys()),
                    hidden_dropout=self.dropout
                ).to(self.device)   
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.lr)
        scaler = torch.amp.GradScaler(self.device)

        model.train()

        
        #Building datasets

        for epoch in range(self.epochs):

            total_losses = 0
            all_preds = []
            all_labels = []
                        
            for fields, b_labels in train_dataloader:

                fields = {k: v.to(self.device) for k, v in fields.items()}
                b_labels = b_labels.to(self.device)

                optimizer.zero_grad()

                #logits = model(fields) #predictions
                #loss = self.criterion(logits, b_labels)

                with torch.autocast(device_type=self.device.type):
                    logits = model(fields)
                    loss = self.criterion(logits, b_labels)
                #loss.backward()
                #optimizer.step()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                total_losses += loss.item()
                preds = torch.argmax(logits, dim=1)
                
                all_preds.extend(preds.cpu().tolist())
                all_labels.extend(b_labels.cpu().tolist())
            
            val_acc, val_prec, val_rec, val_f1 = self.evaluate(model, val_data, label_cl)
            #accuracy per epoch
            accuracy = accuracy_score(all_labels, all_preds)
            prec, rec, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='macro', zero_division=0)
            
            print(f"Epoch: {epoch + 1} \n\
                  Accuracy: {accuracy} \n \
                  Precision: {prec} \n \
                  Recall: {rec} \n \
                  Macro F1-score {f1}")

            print(f"--------Validation Scores------ \n \
                    V Accuracy: {val_acc} \n \
                    V Precision: {val_prec} \n \
                    V Recall: {val_rec} \n \
                    V Macro F1-score: {val_f1}\n ")
            #correct = sum(pred == label for pred, label in zip(all_preds, all_labels))
            #accuracy = correct / len(all_labels)

            #print(f"Epoch {epoch + 1}/ {self.epochs}, loss: {total_losses:.4f}, accuracy: {accuracy:.4f}")
        
        #return model

    def evaluate(self, model, val_data, label_cl):
        model.eval()
        total_losses = 0
        all_preds = []
        all_labels = []
        
        val_fold = DataProcessor(val_data)
        val_fold.label_extractor()

        val_dataloader = DataLoader(
                val_fold,
                batch_size =    self.batch_size,
                shuffle =       True,
                num_workers =   4,
                pin_memory =    False
                )
        with torch.no_grad():
            for fields, b_labels in val_dataloader:
                fields = {k: v.to(self.device) for k,v in fields.items()}
                b_labels = b_labels.to(self.device)

                with torch.autocast(device_type=self.device.type):
                    logits = model(fields)
                    loss = self.criterion(logits, b_labels)

                total_losses += loss.item()
                preds = torch.argmax(logits, dim = 1)

                all_preds.extend(preds.cpu().tolist())
                all_labels.extend(b_labels.cpu().tolist())
        accuracy = accuracy_score(all_labels, all_preds)
        prec, rec, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='macro', zero_division = 0)

        return accuracy, prec, rec, f1
        #with torch.no_grad():
           # for fields, b_labels in d




