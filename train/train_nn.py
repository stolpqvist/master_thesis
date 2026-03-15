from preprocessing.pre_nn import SPTokenizer
from model.rnn import RNN

from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset
import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import precision_recall_fscore_support ,accuracy_score

class NNTrain:
    def __init__(self, lr, epochs, batch_size, dropout, hidden_size):
        self.lr = lr
        self.n_epochs = epochs
        self.batch_size = batch_size
        self.dropout = dropout
        self.hidden_size = hidden_size
        self.criterion = nn.CrossEntropyLoss()

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")



    def training_loop(self, train_data, val_data):
        
        spt = SPTokenizer(model="tokenizer")
        
        #extract labels
        spt.label_extractor(train_data)

        print("Starting tokenizing")

        num_classes = len(spt.label2id.keys())
        t_tokens, t_labels = spt.tokenizer(train_data)
        v_t_tokens, v_t_labels = spt.tokenizer(val_data)

        print("Tokenizing completed")
        
        train_fold = TensorDataset(t_tokens, t_labels)
        val_fold = TensorDataset(v_t_tokens, v_t_labels)
        
        train_dataloader = DataLoader(
            train_fold,
            batch_size= self.batch_size,
            shuffle=    True
        )
        val_dataloader = DataLoader(
            val_fold,
            batch_size= self.batch_size,
            shuffle=    True
        )


        model = RNN(input_size=spt.model.get_piece_size(), 
                    hidden_size=self.hidden_size, 
                    num_classes=num_classes, dropout=self.dropout).to(self.device) 

        optimmizer = torch.optim.AdamW(model.parameters(), lr=self.lr, weight_decay=0.1)

        model.train()

        #Starting training 10 epochs per 1 fold -> then repeat 10 times (10 folds)
        #Here -> saving the best model per epoch

        best_val_f1 = 0

        best_acc = 0
        best_rec = 0
        best_prec = 0

        for epoch in tqdm(range(self.n_epochs)):

            print("Starting training")

            total_losses = []
            all_preds = []
            all_labels = []

            for tokens, labels in train_dataloader:
                tokens = tokens.to(self.device)
                labels = labels.to(self.device)

                optimmizer.zero_grad()

                logits = model(tokens)
                loss = self.criterion(logits, labels)

                loss.backward()
                optimmizer.step()

                total_losses.append(loss.item())

                preds = torch.argmax(logits, dim=1)

                all_preds.append(preds)
                all_labels.append(labels)
            
            all_preds = torch.cat(all_preds).cpu().numpy()
            all_labels = torch.cat(all_labels).cpu().numpy()

            train_loss = sum(total_losses)/ len(total_losses)
            acc = accuracy_score(all_labels, all_preds)
            prec, rec, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='macro', zero_division=0)

            print(f"Epoch {epoch + 1}: Train loss = {train_loss:.4f}, \
                  Accuracy: {acc}, \
                   Precision: {prec}, \
                    Recall: {rec}, \
                    F1-Score: {f1}" 
                    )    
        
            val_acc, val_prec, val_rec, val_f1 = self.evaluate(val_dataloader, model)

            print(f"Validation results: \
                    Val Accuracy: {val_acc}, \
                    Val Precision: {val_prec}, \
                    Val Recall: {val_rec}, \
                    Val F1-Score: {val_f1}" 
                    )
        
        #early stopping for 1 fold
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                epoch = epoch
                best_acc = val_acc
                best_prec = val_prec
                best_rec = val_rec
            else:
                print("We are returning")
                return best_val_f1, best_acc, best_prec, best_rec, epoch
                

    def evaluate(self, v_loader, model):

        model.eval()

        total_losses = []
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for tokens, labels in v_loader:
                #print("We are at prediction")
                tokens = tokens.to(self.device)
                labels = labels.to(self.device)

                logits = model(tokens)

                loss = self.criterion(logits, labels)

                total_losses.append(loss.item())
                

                preds = torch.argmax(logits, dim=1)

                all_preds.append(preds)
                all_labels.append(labels)

        print("We have predicted")

        all_preds = torch.cat(all_preds).cpu().numpy()
        all_labels = torch.cat(all_labels).cpu().numpy()

        val_loss = sum(total_losses)/ len(total_losses)
        acc = accuracy_score(all_labels, all_preds)
        prec, rec, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='macro', zero_division=0)

        #print(f"Validation results: Val loss = {val_loss:.4f}, \
        #        Val Accuracy: {acc}, \
        #        Val Precision: {prec}, \
        #        Val Recall: {rec}, \
        #        Val F1-Score: {f1}" 
        #        )
        
        return acc, prec, rec, f1
                