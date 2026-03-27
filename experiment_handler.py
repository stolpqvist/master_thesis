#TODO To make this class be able to handle and coordinate models
#TODO To handle the different types of trainers
#TODO To handle file writing and plot creation
#TODO To handle parameter hunting for various models

import pandas as pd
from utils.path_manager import PathManager
import torch
from config import Config

#TODO NEW:  3. weight decay number for RoBERTa 

class ExperimentOrganiser:

    def __init__(self, df: pd.DataFrame, config: Config):#model_name, bg, columns, label, lr, dropout, epochs, batch_size, create_data=False, param_hunt=False, train=False, test=False):
        self.df = df
        self.model = config.model.value
        self.bg = config.bg
        self.columns = config.columns
        self.label = config.label
        self.lr = config.lr
        self.dropout = config.dropout
        self.epochs = config.n_epochs
        self.batch_size = config.batch_size

        self.create_data = config.create_data
        self.ph = config.param_hunt #False/True
        self.train = config.train
        self.test = config.test

        self.organiser()
    
    def organiser(self):
        print(f"MODEL IS: repr={repr(self.model)}") 

        if self.create_data:
        
            self.create_datasets(self.df, self.columns, self.label)

            #create the tokenizer for NN on full dataset
            from preprocessing.pre_nn import SPTokenizer

            spt = SPTokenizer(self.columns, self.label, self.df)
            

        if self.train:

            model, f1,  acc, prec, rec, epoch = self.train_setup(
                                                                self.bg, 
                                                                self.columns, 
                                                                self.label, 
                                                                self.lr, 
                                                                self.dropout, 
                                                                self.epochs, 
                                                                self.batch_size)
            self.save_model(model, file=None)
        
        if self.ph:

            self.param_hunt(
                            self.bg, 
                            self.columns, 
                            self.label,   
                            self.epochs, 
                            self.batch_size)
        
        if self.test:

            #load the model from directory

            acc, prec, rec, f1 = self.evaluate(
                self.df,
                self.bg, 
                self.columns, 
                self.label, 
                self.lr, 
                self.dropout, 
                self.epochs, 
                self.batch_size,
                model
            )


            
        
        
    
    def train_setup(self, bg, columns, label, lr, dropout, epochs, batch_size):
        from data_handling.strat_fold import StratifiedFold
        

        #file = f"datasets/{bg}/{bg}_trainval.csv"

        #df = pd.read_csv(file, usecols=[columns])

            
        sfold = StratifiedFold(k=10)
        sfold.stratifier(self.df, label)

        fold_f1s = []
        fold_acc = []
        fold_prec = []
        fold_rec = []

        for train_ids, val_ids in sfold:
            
            train_fold=self.df.iloc[train_ids]
            val_fold=self.df.iloc[val_ids]

            if self.model != 'roberta':
                print(self.model)
                from train.train_nn import NNTrain

                trainer = NNTrain(
                    model=self.model,
                    lr=lr,
                    epochs=epochs,
                    batch_size=batch_size,
                    dropout=dropout,
                    hidden_size=512,
                    columns=columns,
                    label=label
                )


            else:
                from train.train import ModelTrain
                import copy

                print("Starting Roberta training")
                
                trainer = ModelTrain(
                    columns=self.columns,
                    label=self.label,
                    lr=lr,
                    n_epochs=epochs,
                    batch_size = batch_size,
                    dropout= dropout,
                    weight_decay=1e-4 #CHECK WHICH NUMBER 
                    )

            model, f1,  acc, prec, rec, epoch = trainer.training_loop(train_fold, val_fold)

            fold_f1s.append(f1)
            fold_acc.append(acc)
            fold_prec.append(prec)
            fold_rec.append(rec)
            
            del trainer


        mean_f1 = sum(fold_f1s) / len(fold_f1s) #mean f1
        mean_acc = sum(fold_acc) / len(fold_acc)
        mean_prec = sum(fold_prec) / len(fold_prec)
        mean_rec = sum(fold_rec) / len(fold_rec)

        with open(f"Results{self.model}_{bg}.txt", 'a') as r_file:
                    r_file.write(f"Model, Dropout: {dropout}, LR: {lr}, Epochs: {epoch}, F1-Score: {mean_f1}, Accuracy: {mean_acc}, Precision: {mean_prec}, Recall: {mean_rec}")
        
        return model, f1, acc, prec, rec, epoch
        


    def evaluate(self, val_data, bg, columns, label, lr, dropout, epochs, batch_size, model):

        if self.model != 'roberta':
            from .train.train_nn import NNTrain

            trainer = NNTrain(
                lr=         lr,
                epochs=     epochs,
                batch_size= batch_size,
                dropout=    dropout,
                hidden_size=512,
                columns=    columns,
                label=      label
            )

        else:
            from .train.train import ModelTrain
            import copy
            
            trainer = ModelTrain(
                columns=    self.columns,
                label=      self.label,
                lr=         lr,
                n_epochs=   epochs,
                batch_size= batch_size,
                dropout=    dropout
                )
        
        accuracy, prec, rec, f1 = trainer.evaluate(val_data, model)
        return


    def save_model(self, model, file=None):
        if file is None:
            file = f"model/{self.model}/{self.model}.pt"
        torch.save(model.state_dict(), file)


    def create_datasets(self, df, columns, label):

        from create_datasets.split_dataset import GroupSplit
        from sklearn.model_selection import  train_test_split

        pm = PathManager("./")
        pm.setup()
        #print("Before GroupSlit")
        gs = GroupSplit(pm, columns, label, df)

        for bg in gs.groups: #iterate over group names

            df = pd.read_csv(pm.get_dataset_csv(bg))

            df_trainval, df_test = train_test_split(
                df,
                test_size=0.1,
                random_state=42,
                stratify=df[label]
            )

            df_trainval = df_trainval.reset_index(drop=True)
            df_test     = df_test.reset_index(drop=True)

            print(f" {bg} Dataset sizes: train+val: {len(df_trainval)},  test: {len(df_test)}")

            df_trainval.to_csv(pm.get_trainval_csv(bg), index=False)
            df_test.to_csv(pm.get_test_csv(bg), index=False)

        print("Datasets created.")

    def param_hunt(self, bg, columns, label, epochs, batch_size):

        import optuna 
        

        def objective(trial):

            if self.model != 'roberta':
                lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
                dropout = trial.suggest_float("dropout", 0.2, 0.5)

            else:
                lr = trial.suggest_float("lr", 1e-6, 3e-5, log=True)
                dropout = trial.suggest_float("dropout", 0.1, 0.4)
                weight_decay = trial.suggest_float("weight_decay", 1e-4, 1e-1, log=True)

            #fold_f1s = []


            model, f1,  acc, prec, rec, epoch = self.train_setup(
                        bg=bg,
                        columns=    columns,
                        label=      label,
                        lr=         lr,
                        dropout=    dropout,
                        epochs=     epochs,
                        batch_size= batch_size
                        ) 

            #fold_f1s.append(f1)
              
                
            del trainer


            return f1 #sum(fold_f1s) / len(fold_f1s)


        
        def save_trial(study, trial):
        
            with open(f"Results{self.model}_{bg}.txt", 'a') as r_file:
                r_file.write(
                    f"Trial {trial.number} | F1: {trial.value:.4f} |"
                    f"LR {trial.params['lr']:.6f} | Dropout: {trial.params['dropout']:.4f}\n"
                )

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials = 25, callbacks=[save_trial]) #25 combinations


        #Here just write the best param
        with open(f"Results{self.model}_{bg}.txt", 'a') as r_file:
            r_file.write(
                f"\n Best LR: {study.best_params['lr']:.6f} |"
                f"Dropout: {study.best_params['dropout']:.4f}| "
                f"F1: {study.best_value:.4f}\n"
            )         

        
