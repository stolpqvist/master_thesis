#TODO To make this class be able to handle and coordinate models
#TODO To handle the different types of trainers
#TODO To handle file writing and plot creation
#TODO To handle parameter hunting for various models
#
import pandas as pd
from utils.path_manager import PathManager
import torch


class ExperimentOrganiser:

    def __init__(self, *args, **kwargs):
        self.model = model
        if self.model.lower() in ['roberta', 'rnn', 'cnn']:
            if self.model.lower() != 'roberta':
                from .preprocessing.pre_nn import 
                trainer =
        else:
            print('Not model available for selection')
            break
    
    def load_model(self, model):
        raise NotImplementedError
    
    def load_trainer(self):

        if self.model != 'roberta':
            load = x
        else:
            NotImplementedError

    
    def train_setup(self, bg, columns, label, lr, dropout, epochs, batch_size):
        from .data_handling.strat_fold import StratifiedFold
        

        file = f"datasets/{bg}/{bg}_trainval.csv"

        df = pd.read_csv(file, usecols=[columns])

        label_cl = label
            
        sfold = StratifiedFold(k=10)
        sfold.stratifier(df, label_cl)

        fold_f1s = []

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
                    hidden_size=512
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

            f1,  acc, prec, rec, epoch = trainer.training_loop(train_fold, val_fold)
            
            del trainer

        return sum(fold_f1s) / len(fold_f1s) #mean f1



    def evaluate(self):
        raise NotImplementedError


    def create_datasets(self, filepath, columns, label):

        from .create_datasets.split_dataset import GroupSplit
        from sklearn.model_selection import  train_test_split

        pm = PathManager("./")
        pm.setup()

        gs = GroupSplit(pm, columns, label, filepath)

        for bg in gs.groups: #iterate over group names
            df = pd.read_csv(pm.get_dataset_csv(bg))

            label_cl = label #'TilldeladBeredningsgruppKortNamn'

            df_trainval, df_test = train_test_split(
                df,
                test_size=0.1,
                random_state=42,
                stratify=df[label_cl]
            )

            df_trainval = df_trainval.reset_index(drop=True)
            df_test     = df_test.reset_index(drop=True)

            print(f" {bg} Dataset sizes: train+val: {len(df_trainval)},  test: {len(df_test)}")

            df_trainval.to_csv(pm.get_trainval_csv(bg), index=False)
            df_test.to_csv(pm.get_test_csv(bg), index=False)

        print("Datasets created.")
