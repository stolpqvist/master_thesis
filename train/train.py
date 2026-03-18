from preprocessing.pre_roberta import DataProcessor
from model.roberta import CustomXLMRoberta
from torch.utils.data import DataLoader, TensorDataset
import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
from transformers import get_cosine_schedule_with_warmup
import copy

class ModelTrain:
    def __init__(self, lr, n_epochs, batch_size, dropout, weight_decay):

        self.lr = lr
        self.epochs = n_epochs
        self.batch_size = batch_size
        self.criterion = nn.CrossEntropyLoss()
        self.dropout = dropout
        self.weight_decay = weight_decay
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")



    def training_loop(self, data, val_data):
        #TODO Implement random parameter generation ✅
        #TODO Implement writing to file ✅
        #TODO Implement constructor that constructs folders as needed
        #TODO Implement early stopping to reduce training time ✅
        #TODO Implement model saving  ✅
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
            num_workers=3,
            pin_memory=False
        )

        print("Data loaded successfully")

        #Init the model

        model = CustomXLMRoberta(
                    num_classes=len(train_fold.label2id.keys()),
                    hidden_dropout=self.dropout
                ).to(self.device)   
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scaler = torch.amp.GradScaler(self.device)


        #Warmup for the model 

        total_steps = len(train_dataloader) * self.epochs
        warmup_steps = int(total_steps * 0.06)


        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )

        model.train()

        
        #STarting training 10 epochs per 1 fold -> then repeat 10 times (10 folds)
        #Here -> saving the best model per epoch
        best_val_f1 = 0
        #best_model = None

        best_acc = 0
        best_rec = 0
        best_prec = 0


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

                #for warmup ONLY
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scale_before = scaler.get_scale()
                #############

                scaler.step(optimizer)
                scaler.update()
                
                #this also for Warmup ONLY
                #####################
                if scale_before <= scaler.get_scale():
                    scheduler.step()
                ####################

                total_losses += loss.item()
                preds = torch.argmax(logits, dim=1)
                
                all_preds.extend(preds.cpu().tolist())
                all_labels.extend(b_labels.cpu().tolist())
            
            val_acc, val_prec, val_rec, val_f1 = self.evaluate(model, val_data)
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
            
            #early stopping for 1 fold
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                #best_model = copy.deepcopy(model.state_dict())
                epoch = epoch
                best_acc = val_acc
                best_prec = val_prec
                best_rec = val_rec
            else:
                return best_val_f1, best_acc, best_prec, best_rec, epoch
            
        


    def evaluate(self, model, val_data):
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
      



