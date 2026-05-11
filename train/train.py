"""
This module handles the training, evaluation, and testing of the RoBERTa model.
"""
from preprocessing.pre_roberta import DataProcessor
from model.roberta import CustomXLMRoberta
from torch.utils.data import DataLoader, TensorDataset
import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
from transformers import get_cosine_schedule_with_warmup
import copy
import pandas as pd

class ModelTrain:
    """
    This class handles the training, evaluation, and testing of the RoBERTa model.

    Arguments:
        columns (list(str)) = The headers of the columns the be used for informating in training, evaluation and testing.
        label (str) = The label header to be used to compared model predictions against.
        lr (float) = The learning rate value to be applied.
        epochs (int) = The number of epochs to train over.
        batch_size (int) = The number of batches to split the data into for batch loading.
        criterion (nn.CrossEntropyLoss) = The loss function to be used.
        dropout (float) = The dropout rate to be applied for the model.
        weight_decay (float) = The amount of weight decay to be applied to the model.
        device (torch.device(cuda|mps|cpu)) = The device to be trained on.

    Methods:
        training_loop(data: pd.DataFrame, val_data:pd.DataFrame) -> nn.Module, float, float, float, float, int
            Deals with the training loop for the model.

        evaluate(model: str|nn.Module, val_data: pd.DataFrame, bg: str,boot: bool)
            Handles the evaluation of the model.
    


    """
    def __init__(self, columns: list[str], label:str, lr:float, n_epochs:int, batch_size:int, dropout:float, weight_decay:float=1e-4):
        
        self.columns = columns
        self.label = label

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



    def training_loop(self, data: pd.DataFrame, val_data:pd.DataFrame) -> tuple[nn.Module, float, float, float, float, int]:
        """
        This method handles the training and the subsequent evaluation of the model during training. It trains on K - 1 folds and
        validates on 1 fold.

        Arguments:
            data (pd.DataFrame) = The training data to train the model on.
            val_data (pd.DataFrame) = The validation data to evaluate the model on during training.
        
        Returns:
            best_model (nn.Module) = The best performing model for a particular fold.
            best_val_f1 (float) = The best F1-score achieved during a particular fold.
            best_acc (float) = The best accuracy value achieved during a particular fold.
            best_prec (float) = The best precision value achieved during a particular fold.
            best_rec (float) = The best recall value achieved during a particular fold.
            epoch (int) = The epoch in which the best scores were achieved.
        """
        print(f"Training on: {self.device}")

        #Prepare the data:
        train_fold = DataProcessor(data, self.columns, self.label)
        train_fold.label_extractor()

        train_dataloader = DataLoader(
            train_fold,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,
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
        best_model = None

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
                return best_model, best_val_f1, best_acc, best_prec, best_rec, epoch
        return best_model,best_val_f1, best_acc, best_prec, best_rec, epoch
            

    def evaluate(self, model: str|nn.Module, val_data: pd.DataFrame, bg = None,boot=False):
        """
        This method handles the evaluation and testing of the RoBERTa model.

        Arguments:
            model (str|nn.Module) = Either the name of the model in string-format or the nn.Module class object.
            val_data (pd.DataFrame) = A pandas dataframe containing the validation data or the testing data.
            bg (str) = The particular subject matter group to be validated/tested on.
            boot (bool) = A boolean that checks whether bootstrapping is being done or not, if true it returns predictions and labels.


        Returns:
            accuracy (float) = The accuracy value for the validation fold or testing.
            prec (float) = The precisions value for the vlidation fold or testing.
            rec (float) = The recall value for the validation fold or testing.
            f1 (float) = The F1-score for the validation fold or testing.
            all_preds (list(int)) = If the condition is met, all the predictions are returned.
            all_labels (list(int)) = If the conditoin is met, all the true labels are returned
      

        """
        total_losses = 0
        all_preds = []
        all_labels = []
        val_fold = DataProcessor(df=val_data, columns=self.columns, label=self.label)
        val_fold.label_extractor()
        
        if boot:
            model_path = model
            model = CustomXLMRoberta(
                    num_classes=len(val_fold.label2id.keys()),
                    hidden_dropout=self.dropout
                ).to(self.device)   
            if bg is not None:
                model_path = f'models/{model_path}/{model_path}_{bg}.pt'
                print(model_path)
            model.load_state_dict(torch.load(model_path, map_location=self.device))
            val_dataloader = DataLoader(
                val_fold,
                batch_size =    self.batch_size,
                shuffle =       False,
                num_workers =   0,
                pin_memory =    False
                )
        
        else:
            val_dataloader = DataLoader(
                    val_fold,
                    batch_size =    self.batch_size,
                    shuffle =       True,
                    num_workers =   0,
                    pin_memory =    False
                    )
        model.eval()
        
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
        if boot:
            return all_preds, all_labels, accuracy, prec, rec, f1
        else:    
            return accuracy, prec, rec, f1
      



