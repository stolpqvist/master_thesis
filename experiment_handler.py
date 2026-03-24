#TODO To make this class be able to handle and coordinate models
#TODO To handle the different types of trainers
#TODO To handle file writing and plot creation
#TODO To handle parameter hunting for various models

import pandas as pd
from utils.path_manager import PathManager
import torch

#TODO NEW:  1. fix the pm class to check and add folder for a model and results
#           2. Organiser
#           3. weight decay number for RoBERTa 

class ExperimentOrganiser:

    def __init__(self, model_name, bg, columns, label, lr, dropout, epochs, batch_size):
        self.model = model_name
        self.bg = bg
        self.columns = columns
        self.label = label
        self.lr = lr
        self.dropout = dropout
        self.epochs = epochs
        self.batch_size = batch_size
    
    def organiser(self):
        model, f1,  acc, prec, rec, epoch = self.train_setup(
                                                            self.bg, 
                                                            self.columns, 
                                                            self.label, 
                                                            self.lr, 
                                                            self.dropout, 
                                                            self.epochs, 
                                                            self.batch_size)
        
        self.save_model(model, file=None)
    
    def train_setup(self, bg, columns, label, lr, dropout, epochs, batch_size):
        from .data_handling.strat_fold import StratifiedFold
        

        file = f"datasets/{bg}/{bg}_trainval.csv"

        df = pd.read_csv(file, usecols=[columns])

            
        sfold = StratifiedFold(k=10)
        sfold.stratifier(df, label)

        fold_f1s = []
        fold_acc = []
        fold_prec = []
        fold_rec = []

        for train_ids, val_ids in sfold:
            
            train_fold=df.iloc[train_ids]
            val_fold=df.iloc[val_ids]

            if self.model != 'roberta':
                from .train.train_nn import NNTrain

                trainer = NNTrain(
                    lr=lr,
                    epochs=epochs,
                    batch_size=batch_size,
                    dropout=dropout,
                    hidden_size=512,
                    columns=columns,
                    label=label
                )


            else:
                from .train.train import ModelTrain
                import copy
                
                trainer = ModelTrain(
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
                lr=lr,
                epochs=epochs,
                batch_size=batch_size,
                dropout=dropout,
                hidden_size=512,
                columns=columns,
                label=label
            )

        else:
            from .train.train import ModelTrain
            import copy
            
            trainer = ModelTrain(
                lr=lr,
                n_epochs=epochs,
                batch_size = batch_size,
                dropout= dropout
                )
        
        trainer.evaluate(val_data, model)


    def save_model(self, model, file=None):
        if file is None:
             file = f"model/{self.model}/{self.model}.pt"
        torch.save(model.state_dict(), file)


    def create_datasets(self, filepath, columns, label):

        from .create_datasets.split_dataset import GroupSplit
        from sklearn.model_selection import  train_test_split

        pm = PathManager("./")
        pm.setup()

        gs = GroupSplit(pm, columns, label, filepath)

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

    def param_hunt(self, bg, columns, label, lr, dropout, epochs, batch_size):

        import optuna 
        



        def objective(trial):

            if self.model != 'roberta':
                lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
                dropout = trial.suggest_float("dropout", 0.2, 0.5)

            else:
                lr = trial.suggest_float("lr", 1e-6, 3e-5, log=True)
                dropout = trial.suggest_float("dropout", 0.1, 0.4)
                weight_decay = trial.suggest_float("weight_decay", 1e-4, 1e-1, log=True)

            fold_f1s = []


            model, f1,  acc, prec, rec, epoch = self.train_setup(
                        bg=bg,
                        columns=    columns,
                        label=      label,
                        lr=         lr,
                        dropout=    dropout,
                        epochs=     epochs,
                        batch_size= batch_size
                        ) 

            fold_f1s.append(f1)
              
                
            del trainer


            return sum(fold_f1s) / len(fold_f1s)


        
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

        
