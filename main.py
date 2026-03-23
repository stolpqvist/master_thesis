import argparse
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import  train_test_split
from data_handling.strat_fold import StratifiedFold

from preprocessing.pre_roberta import DataProcessor
from model.roberta import CustomXLMRoberta
from train.train import ModelTrain
import numpy as np
from scipy import stats
import copy
from utils.path_manager import PathManager
from create_datasets.split_dataset import GroupSplit



def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('-bg', type=str, default='NT')
    parser.add_argument('--create_datasets', '-cd', action='store_true', default=False) 
    parser.add_argument('-md', type=str, default= 'roberta') #model
    parser.add_argument('-dr', type=float, default=0.1)
    parser.add_argument('-lr', type=float, default=0.00001)
    parser.add_argument('-e', type=int, default=5) #epochs
    parser.add_argument('--batch_size', '-b', type=int, default=3)
    parser.add_argument('-tr', action='store_true', default=False)
    parser.add_argument('--param_hunt', '-p', action='store_true', default=False)
    parser.add_argument('-test_size', type=float, default=0.1)       # Test size
    args = parser.parse_args()

    if args.create_datasets:

        #NEW FROM HERE
        pm = PathManager("./")

        #create datset dir
        pm.setup()


        #create group specific dirs in dataset ad save the group specific file

        gs = GroupSplit(pm)

        #READING the file that has already been saved

       #df = pd.read_csv(pm.get_dataset_csv(args.bg)) 
       
        #UNTIL HERE

        for bg in gs.groups: #iterate over group names
            df = pd.read_csv(pm.get_dataset_csv(bg))


        #file = f"../datasets/{args.bg}/{args.bg}_dataset.csv"
        #df = pd.read_csv(file)

            label_cl = 'TilldeladBeredningsgruppKortNamn'

            df_trainval, df_test = train_test_split(
                df,
                test_size=args.test_size,
                random_state=42,
                stratify=df[label_cl]
            )

            df_trainval = df_trainval.reset_index(drop=True)
            df_test     = df_test.reset_index(drop=True)

            print(f" {bg} Dataset sizes: train+val: {len(df_trainval)},  test: {len(df_test)}")

            #INSTEAD OF OLD HERE:
            #SAVE THE NEW FILES "".to_csv"

            df_trainval.to_csv(pm.get_trainval_csv(bg), index=False)
            df_test.to_csv(pm.get_test_csv(bg), index=False)
            #FINISH NEW
        
        #saving datasets Train/Test
        #trainval = f"../datasets/{args.bg}/{args.bg}_trainval.csv"
        #df_trainval.to_csv(trainval, index=False)

        #test = f"../datasets/{args.bg}/{args.bg}_test.csv"
        #df_test.to_csv(test, index=False)

        print("Datasets created.")


    if args.tr:
        
        file = f"datasets/{args.bg}/{args.bg}_trainval.csv"

        #df = pd.read_csv(pm.get_trainval_csv(args.bg)...)
        #for test:
        #df = pd.read_csv(pm.get_test_csv(args.bg)...)

        df = pd.read_csv(file, usecols=[
            "TilldeladBeredningsgruppKortNamn",
            "AnsökanTitel",
            "AnsökanTitelEng",
            "Beskrivning",
            "Nyckelord"
        ])

        label_cl = 'TilldeladBeredningsgruppKortNamn'
        
        sfold = StratifiedFold(k=10)
        sfold.stratifier(df, label_cl)

        best_f1_from_all_folds = 0
        best_model_from_all_folds = None #because we will test all parameters and have the best model for EACH parameter
    
        #to get the best model per fold -> we need to compare all models from 10 epochs

        #train/ val indices - Training loop 9/1 - Repeat (K=10)
        for train_ids, val_ids in sfold:

            train_fold=df.iloc[train_ids]
            val_fold=df.iloc[val_ids]

            
            trainer = ModelTrain(
                lr=args.lr,
                n_epochs=args.e,
                batch_size = args.batch_size,
                dropout= args.dr
                )

            model, f1,  acc, prec, rec, epoch = trainer.training_loop(train_fold, val_fold, label_cl)

            #Tracking the best model from 10 folds:
            if f1 > best_f1_from_all_folds:
                best_f1_from_all_folds = f1
                best_model_from_all_folds = best_model_state = copy.deepcopy(model.state_dict())
                best_acc = acc
                best_prec = prec
                best_rec = rec
                epoch = epoch
    
            
            del trainer


        #Saving the best model from all 10 folds per parameter 
        torch.save(best_model_from_all_folds.state_dict(), './Best_model.pt')


        #Save results into the file
        with open('Results.txt', 'a') as r_file:
            r_file.write(f"Model, Dropout: {args.dr}, LR: {args.lr}, Epochs: {epoch}, Total N epochs: {args.e}, F1-Score: {best_f1_from_all_folds}, Accuracy: {best_acc}, Precision: {best_prec}, Recall: {best_rec}")
        
    if args.param_hunt:

        import optuna

        #lrs = stats.loguniform(1e-6, 3e-5).rvs(10)
        #dropouts = stats.uniform(0.1, 0.4).rvs(10)

        #hyper_parameters = {
        #    "lrs":      lrs,
        #    "dropouts": dropouts,
        #}

        file = f"datasets/{args.bg}/{args.bg}_trainval.csv"
        df = pd.read_csv(file, usecols=[
            "TilldeladBeredningsgruppKortNamn",
            "AnsökanTitel",
            "AnsökanTitelEng",
            "Sammanfattning",
            "Populärbeskrivning",
            "Nyckelord"
        ])

        label_cl = 'TilldeladBeredningsgruppKortNamn'
        
        sfold = StratifiedFold(k=5)
        sfold.stratifier(df, label_cl)


        def objective(trial):

            lr = trial.suggest_float("lr", 1e-6, 3e-5, log=True)
            dropout = trial.suggest_float("dropout", 0.1, 0.4)
            weight_decay = trial.suggest_float("weight_decay", 1e-4, 1e-1, log=True)

            fold_f1s = []

            for train_ids, val_ids in sfold:

                train_fold=df.iloc[train_ids]
                val_fold=df.iloc[val_ids]


                trainer = ModelTrain(
                            lr=lr,
                            n_epochs=args.e,
                            batch_size = args.batch_size,
                            dropout= dropout,
                            weight_decay=weight_decay
                            )
                
                f1,  acc, prec, rec, epoch = trainer.training_loop(train_fold, val_fold, label_cl)
                fold_f1s.append(f1) #validation f1s

                del trainer
            
            return sum(fold_f1s) / len(fold_f1s) #mean across all folds
        
        def save_trial(study, trial):
        
            with open(f'Results_roberta_{args.bg}.txt', 'a') as r_file:
                r_file.write(
                    f"Trial {trial.number} | F1: {trial.value:.4f} |"
                    f"LR {trial.params['lr']:.6f} | Dropout: {trial.params['dropout']:.4f}\n"
                )

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials = 25, callbacks=[save_trial]) #25 combinations


        #Here just write the best param
        with open(f'Results_roberta_{args.bg}.txt', 'a') as r_file:
            r_file.write(
                f"\n Best LR: {study.best_params['lr']:.6f} |"
                f"Dropout: {study.best_params['dropout']:.4f}| "
                f"F1: {study.best_value:.4f}\n"
            )             


        #best_f1_from_all_folds = 0
        #best_model_from_all_folds = None #because we will test all parameters and have the best model for EACH parameter
        

        #for lr in hyper_parameters["lrs"]:
        #    for dropout in hyper_parameters["dropouts"]:
        #        try:
                    #to get the best model per fold -> we need to compare all models from 10 epochs

                    #train/ val indices - Training loop 9/1 - Repeat (K=10)
        #            for train_ids, val_ids in sfold:

        #                train_fold=df.iloc[train_ids]
        #                val_fold=df.iloc[val_ids]

                        
        #                trainer = ModelTrain(
        #                    lr=lr,
        #                    n_epochs=args.e,
        #                    batch_size = args.batch_size,
        #                    dropout= dropout
        #                    )

        #                model, f1,  acc, prec, rec, epoch = trainer.training_loop(train_fold, val_fold, label_cl)

                        #Tracking the best model from 10 folds:
        #                if f1 > best_f1_from_all_folds:
        #                    best_f1_from_all_folds = f1
        #                    best_model_from_all_folds = copy.deepcopy(model.state_dict())
        #                    best_acc = acc
        #                    best_prec = prec
        #                   best_rec = rec
        #                    epoch = epoch
                
                        
        #                del trainer


                    #Saving the best model from all 10 folds per parameter 
        #            torch.save(best_model_from_all_folds.state_dict(), './Best_model.pt')


                    #Save results into the file
        #            with open('Results.txt', 'a') as r_file:
        #                r_file.write(f"Model, Dropout: {dropout}, LR: {lr}, Epochs: {epoch}, Total N epochs: {args.e}, F1-Score: {best_f1_from_all_folds}, Accuracy: {best_acc}, Precision: {best_prec}, Recall: {best_rec} \n")

        #        except Exception as e:
        #            with open('Results.txt', 'a') as r_file:
        #               r_file.write(f"Model, Dropout: {dropout}, LR: {lr}, ERROR {e} \n")









if __name__ == "__main__":
    main()




