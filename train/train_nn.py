from preprocessing.pre_nn import SPTokenizer
from model.rnn import RNN

from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset
import torch
import torch.nn as nn
import numpy as np
import os
from sklearn.metrics import precision_recall_fscore_support ,accuracy_score

class NNTrain:
    def __init__(self, model, lr, epochs, batch_size, dropout, hidden_size, columns, label):
        self.lr = lr
        self.n_epochs = epochs
        self.batch_size = batch_size
        self.dropout = dropout
        self.hidden_size = hidden_size
        self.columns = columns
        self.label = label
        self.label2id: dict | None = None
        self.id2label: dict | None = None
        
        self.model_class = self.get_model(model)
        self.patience = 3
        self.patience_count = 0
        self.criterion = nn.CrossEntropyLoss()
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

    def get_model(self, model):
        if model == 'cnn':
            from model.cnn import ClassificationCNN
            model_class = ClassificationCNN
        else:
            from model.rnn import RNN
            model_class = RNN
        return model_class
       



    def training_loop(self, train_data, val_data):
        print("This is in training looop", self.columns)
        spt = SPTokenizer(self.columns, self.label, model="tokenizer")
        
        #extract labels
        spt.label_extractor(train_data)
        #save to use later
        self.label2id = spt.label2id

        print("Starting tokenizing")

        num_classes = len(spt.label2id.keys())
        t_tokens, t_labels = spt.tokenizer(train_data)
        # v_t_tokens, v_t_labels = spt.tokenizer(val_data)

        print("Tokenizing completed")
        
        train_fold = TensorDataset(t_tokens, t_labels)
        # val_fold = TensorDataset(v_t_tokens, v_t_labels)
        
        train_dataloader = DataLoader(
            train_fold,
            batch_size= self.batch_size,
            shuffle=    True
        )
        #val_dataloader = DataLoader(
        #    val_fold,
            # batch_size= self.batch_size,
            # shuffle=    True
        # )


        model = self.model_class(input_size=    spt.model.get_piece_size(),
                                 #embedding_dim= embedding_dim,
                                 hidden_size=   self.hidden_size, 
                                 num_classes=   num_classes,
                                 dropout=       self.dropout
                                 ).to(self.device) 

        optimmizer = torch.optim.AdamW(model.parameters(), lr=self.lr, weight_decay=0.1)


        #Starting training 10 epochs per 1 fold -> then repeat 10 times (10 folds)
        best_model = None
        best_val_f1 = 0

        best_acc = 0
        best_rec = 0
        best_prec = 0

        print("Starting training")

        for epoch in tqdm(range(self.n_epochs)):
            model.train()

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
        
            val_acc, val_prec, val_rec, val_f1 = self.evaluate(val_data, model)

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
                best_model = model
            else:
                if self.patience_count < self.patience:
                    self.patience_count += 1
                    #print("We are returning")
                else:
                    return best_model,best_val_f1, best_acc, best_prec, best_rec, epoch

        return best_model,best_val_f1, best_acc, best_prec, best_rec, epoch    

    def evaluate(self, val_data, model, boot=False):
        spt = SPTokenizer(self.columns, self.label, model="tokenizer")
        print("This is model in evaluate", model)
        #if hasattr(self, 'label2id') and hasattr(self, 'id2label'):
        if self.label2id is not None and self.id2label is not None:
            spt.label2id = self.label2id  #reuse it
            spt.id2label = self.id2label
        else:
            spt.label_extractor(val_data)
            self.label2id = spt.label2id
            self.id2label = spt.id2label
            
        if isinstance(model, str) and model.endswith('pt'):

            model_path = model
            num_classes = len(spt.label2id.keys())

            #model_name = model.split('/')[-1].split('_')[0] #does not work for windows
            model_name = os.path.basename(model).split("_")[0] 
            print("this is model name", model_name)
            model_class = self.get_model(model_name)
            model = model_class(input_size=    spt.model.get_piece_size(),
                                 hidden_size=   self.hidden_size, 
                                 num_classes=   num_classes,
                                 dropout=       self.dropout
                                 ).to(self.device) 
            model.load_state_dict(torch.load(model_path, map_location=self.device)) #here only for mac map_location=self.device

        v_t_tokens, v_t_labels = spt.tokenizer(val_data)

        print("Val Tokenizing completed")
        
        val_fold = TensorDataset(v_t_tokens, v_t_labels)
        
        v_loader = DataLoader(
            val_fold,
            batch_size= self.batch_size,
            shuffle=    not boot #not shuffle for boot
        )

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
        if boot:
            return all_preds, all_labels, acc, prec, rec, f1
        else:
            return acc, prec, rec, f1
                
