"""
This module handles the organising of tieing together varies modules in order to test, train, evaluate, bootstrap, visualise 
and to do hyperparameter optimisation.
"""
import pandas as pd
from utils.path_manager import PathManager
import torch
from config import Config
from functools import partial

class ExperimentOrganiser:
    """This class handles the organising of experiments by tieing together different modules for training, testing, bootstrapping,
       and visualising. It takes the input from the config class and assigns the various instance variables accordingly.

        Attributes:
            bg (str) = The group that will be used for classifiction, i.e the subject matter group.
            columns (list(str) = The columns from a pandas DataFrame that will be used as training material for the models.
            label (str) = The label column that will be compared against the model's prediction.
            lr (float) = The learning rate as a float value.
            dropout (float) = The dropout rate as a float.
            epochs (int) = The number of epochs for the model to train on.
            batch_size (int) = The number of batches to split the data into.
            k (int) = The number of splits into the dataset for a k-fold cross-evaluation.
            create_datasets (bool) = Whether to create a new sets of data or not.
            ph (bool) = Whether or not to do hyperparameter optimisation.
            train (bool) = Whether to train a model or not.
            test (bool) = Whether to test the model.
            boot (bool) = Whether to do a bootstrapping test, and to do visualisation and to compare multiple models against eachother.
            visual (bool) = Whether to visualise the results in a confusion matrix and boxplot.
            emissions (bool) = Whether to track emissions or not.
      
      Methods:
        organiser(self) -> None
            Organises what to be done, whether testing, training, bootstrapping.

        train_setup(bg: str, columns: list[str], label: str, lr: float, dropout: float, epochs: int, batch_size: int, 
        ph: bool, emissions: bool) 
            -> nn.Module, float, float, float, float, int
                Sets up training for the particular model retrieved from user input.

        evaluate(self, model: nn.Module, val_data:pd.DataFrame, bg: str, columns: list[str], label: str, lr: str,
        dropout: str, epochs: int, batch_size: int, boot: bool)
            -> list, list, float, float, float, float | float, float, float, float
                Either evaluates a particular fold or evaluates on the test set.

        save_model(model:nn.Module, file: str)-> None
            Saves the model.

        create_datasets(df: pd.DataFrame, columns: list[str], label: str) -> None
            Creates the necessary datasets for training, evaluation and testing for all of the groups.

        param_hunt(bg: str, columns: list[str], label: str, epochs: int, batch_size: int, emission: bool) -> None
            Handles the hyperparameter optimisation with the optuna library. 
    """

    def __init__(self, df: pd.DataFrame, config: Config):#model_name, bg, columns, label, lr, dropout, epochs, batch_size, create_data=False, param_hunt=False, train=False, test=False):
        self.df = df
        if isinstance(config.model, list):
            self.model_name = [m.value for m in config.model]
        else:
            self.model_name = config.model.value 
        self.bg =           config.bg
        self.columns =      config.columns
        self.label =        config.label
        self.lr =           config.lr
        self.dropout =      config.dropout
        self.epochs =       config.n_epochs
        self.batch_size =   config.batch_size
        self.k =            config.k

        self.create_data =  config.create_data
        self.ph =           config.param_hunt #False/True
        self.train =        config.train
        self.test =         config.test
        self.boot =         config.boot
        self.visual =       config.vis
        self.emissions  =   config.emissions
        pm = PathManager('./')
        pm.setup_model(self.model_name)
 
        self.organiser()
   
    
    def organiser(self) -> None:
        """
        This method deals with organising what to do, if to track emissions, create datasets, to train, to do hyperparameter
        optimisation, to do testing, or to do bootstrapping with or without visualisation.
        
        """
        if self.emissions:
            from codecarbon import EmissionsTracker
            emission_tracker = EmissionsTracker(log_level='critical', tracking_mode='machine')
            emission_tracker.start()

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
                                                                self.batch_size,
                                                                emissions = emission_tracker if self.emissions else None
                                                                )
            self.save_model(model, file=None)
        
        if self.ph:

            self.param_hunt(
                            self.bg, 
                            self.columns, 
                            self.label,   
                            self.epochs, 
                            self.batch_size,
                            emission = emission_tracker if self.emissions else None)
        
        if self.test:

            #load the model from directory

            acc, prec, rec, f1 = self.evaluate(
                val_data=self.df,
                bg=self.bg, 
                columns=self.columns, 
                label=self.label, 
                lr=self.lr, 
                dropout=self.dropout, 
                epochs=self.epochs, 
                batch_size=self.batch_size,
                model=self.model_name,
                boot=False
            )
            print(f"The {self.model_name} model completed the test on {self.bg} dataset! \n \
                    Accuracy: {acc}\n \
                    F1-score: {f1} \n \
                    Recall: {rec} \n \
                    Precision: {prec}")

        if self.boot:

            from sig_test import SigTest
            bound_evaluate = partial(self.evaluate, bg=self.bg, columns=self.columns, label=self.label, lr=self.lr, dropout=self.dropout, epochs=self.epochs, batch_size=self.batch_size)
            boot = SigTest(
                    df = self.df,
                    evaluate = bound_evaluate,
                    models = self.model_name,
                    bg = self.bg,
                    label = self.label
                    )
            
            answer = input("Chance test, Model test, Both? \n (C/M): ")

            model_stats = None
            pairwise_result = None

            if answer.upper() == "C":
                model_stats = boot.chance_test()
            if answer.upper() == "M":
                pairwise_result, model_stats = boot.pairwise_test()
         
        
            if self.visual and model_stats is not None:

                from utils.visualisation import Visual

                vis = Visual(
                    bg=self.bg,
                    model_stats=model_stats,
                    model_preds=boot.model_preds,
                    labels=boot.labels,
                    label_names=self.df[self.label].unique().tolist(),
                    pairwise_result=pairwise_result
                )

                vis.plot_confusion_matrix()
                vis.plot_f1_distribution()
                vis.box_plot_f1()
                #vis.plot_pairwise()

    def train_setup(self, bg: str, columns: list[str], label: str, lr: float, dropout: float, epochs: int, batch_size: int, ph: bool=False, emissions: bool =False) -> Tuple[nn.Module, float, float, float, float, int]:
        """
        This method sets up the training and the subsequent evaluation of the validation fold.

        Args:
        bg (str) = The dataset to be read and processed in terms of subject matter group.
        columns (list(str)) = The columns to be used as training data from each row.
        label (str) = The label column to be used to compare the model predictions against.
        lr (float) = The learning rate for a particular model.
        dropout (float) = The dropout rate for a particular model.
        epochs (int) = The amount of epochs to train a model over.
        batch_size (int) = The amount of batches to split the data over during training.
        ph (bool) = A boolean value that determines whether to do hyperparamter optimsation.
        emissions (bool) = A boolean value that determines whether or not to track emissions.

        Returns:
            model (nn.Module) = The best model for the particular fold.
            f1  (float) = The best validation F1-score for that particular set of folds.
            acc (float) = The best validation accuracy score for that particular set of folds.
            prec (float) = The best validation precision score for that particular set of folds.
            rec (float) = The best validation recall score for that particular set of folds. 
            epoch (int) = The epoch on which the best scores were achieved.

        """
        from data_handling.strat_fold import StratifiedFold
        

        #file = f"datasets/{bg}/{bg}_trainval.csv"

        #df = pd.read_csv(file, usecols=[columns])

            
        sfold = StratifiedFold(k=self.k)
        sfold.stratifier(self.df, label)

        fold_f1s = []
        fold_acc = []
        fold_prec = []
        fold_rec = []

        for train_ids, val_ids in sfold:
            
            train_fold=self.df.iloc[train_ids]
            val_fold=self.df.iloc[val_ids]

            if self.model_name != 'roberta':
                from train.train_nn import NNTrain

                trainer = NNTrain(
                    model=self.model_name,
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
#            torch.save(model.state_dict(), f'models/{self.model_name}/{self.model_name}_{self.bg}.pt')


            fold_f1s.append(f1)
            fold_acc.append(acc)
            fold_prec.append(prec)
            fold_rec.append(rec)
            
            del trainer


        mean_f1 = sum(fold_f1s) / len(fold_f1s) #mean f1
        mean_acc = sum(fold_acc) / len(fold_acc)
        mean_prec = sum(fold_prec) / len(fold_prec)
        mean_rec = sum(fold_rec) / len(fold_rec)
        if not ph:
            if self.emissions:
                emissions = emissions.stop()
                print(f"The {self.model_name} produced emissions on {bg} dataset: \n \
                        Emissions: {emissions}")
            with open(f"results/{self.model_name}/text/Results_{self.model_name}_{bg}.txt", 'a') as r_file:
                r_file.write(f"Model, Dropout: {dropout}, LR: {lr}, Epochs: {epoch}, F1-Score: {mean_f1}, Accuracy: {mean_acc}, Precision: {mean_prec}, Recall: {mean_rec}, Emissions: {emissions if emissions is not None else ''} kg CO2eq \n")

            
        return model, f1, acc, prec, rec, epoch
        


    def evaluate(self, model: nn.Module, val_data:pd.DataFrame, bg: str, columns: list[str], label: str, lr: str, dropout: str, epochs: int, batch_size: int, boot: bool=False) -> Tuple[list, list, float, float, float, float | float, float, float, float]:
        """
        This method handles the evaluation of the validation fold or separate testing.

        Args:
            Model (nn.Module) = The model to be trained
            val_data (pandas.DataFrame) = A pandas dataframe containing the validation data or the testing data to be used.
            bg (str) = The dataset to be read and processed in terms of subject matter group.
            columns (list(str)) = The columns to be used as training data from each row.
            label (str) = The label column to be used to compare the model predictions against.
            lr (float) = The learning rate for a particular model.
            dropout (float) = The dropout rate for a particular model.
            epochs (int) = The amount of epochs to train a model over.
            batch_size (int) = The amount of batches to split the data over during training.
            boot (bool) = A boolean that determines whether or not to do bootstrapping.

        Returns:
            When bootstrapping:
                all_preds (list) = The list of predictions from the model.
                all_labels (list) = The list of true labels.
                acc (float) = The accuracy value achieved during evaluation.
                prec (float) = The precision vlaue achieved during evaluation.
                rec (float) = The recall value achieved during evaluation.               
                f1 = The F1-score achieved during evaluation.
            else:
                acc (float) = The accuracy value achieved during evaluation.
                prec (float) = The precision vlaue achieved during evaluation.
                rec (float) = The recall value achieved during evaluation.
                f1 (float) = The F1-score achieved during evaluation.

        """
        import os
        current_model = os.path.basename(model).split('_')[0]
        if current_model != 'roberta':
            from train.train_nn import NNTrain

            trainer = NNTrain(
                model=      current_model,
                lr=         lr,
                epochs=     epochs,
                batch_size= batch_size,
                dropout=    dropout,
                hidden_size=512,
                columns=    columns,
                label=      label
            )

        else:
            from train.train import ModelTrain
            import copy
            
            trainer = ModelTrain(
                columns=    self.columns,
                label=      self.label,
                lr=         lr,
                n_epochs=   epochs,
                batch_size= batch_size,
                dropout=    dropout
                )
        

        if boot:
            all_preds, all_labels, acc, prec, rec, f1 = trainer.evaluate(val_data=val_data, model=model, boot=boot)
            return all_preds, all_labels, acc, prec, rec, f1
        if self.test:
            _, _, acc, prec, rec, f1 = trainer.evaluate(val_data=val_data, model=model, bg=bg, boot=True)
            return acc, prec, rec, f1

        
        else:
            acc, prec, rec, f1 = trainer.evaluate(val_data=val_data, model=model, bg=bg, boot=boot)
            return acc, prec, rec, f1

    def save_model(self, model, file=None) -> None:
        """
        This method handles the saving of a model.

        model = The model that is going to be saved.
        file (str) = The string value representing a file path to where the model should be saved
        """

        if file is None:
            file = f"models/{self.model_name}/{self.model_name}_{self.bg}.pt"
        torch.save(model.state_dict(), file)


    def create_datasets(self, df:pd.DataFrame, columns: list[str], label: str) -> None:
        """
        This method handles the creating of datasets, separates a full dataset into smaller datasets one for each subject matter group.

        df (pd.DataFrame) = The pandas dataframe containing the full dataset to be split.
        columns (list(str)) = A list of columns to be extracted for the smaller datasets.
        label (str) = The label column containing the labels to be compared against the models predictions.


        """
        from create_datasets.split_dataset import GroupSplit
        from sklearn.model_selection import  train_test_split

        pm = PathManager("./")
        pm.setup()
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

    def param_hunt(self, bg: str, columns: list[str], label: str, epochs: int, batch_size: int, emission: bool = None) -> None:
        """
        This method handles the hyperparameter optimisation by leveraging the optuna library.

        Args:
            bg (str) = The dataset to be read and processed in terms of subject matter group.
            columns (list(str)) = The columns to be used as training data from each row.
            label (str) = The label column to be used to compare the model predictions against.
            batch_size (int) = The amount of batches to split the data over during training.
            emission (boolean) = A boolean value that determines whether or not to track emissions.
        """
        import optuna 
        

        def objective(trial):

            if self.model_name != 'roberta':
                lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
                dropout = trial.suggest_float("dropout", 0.2, 0.5)

            else:
                lr = trial.suggest_float("lr", 1e-6, 3e-5, log=True)
                dropout = trial.suggest_float("dropout", 0.1, 0.4)
                weight_decay = trial.suggest_float("weight_decay", 1e-4, 1e-1, log=True)



            model, f1,  acc, prec, rec, epoch = self.train_setup(
                        bg=bg,
                        columns=    columns,
                        label=      label,
                        lr=         lr,
                        dropout=    dropout,
                        epochs=     epochs,
                        batch_size= batch_size,
                        ph =        True,
                        emissions = emission if self.emissions else None
                        ) 
              


            return f1 


        
        def save_trial(study, trial):
        
            with open(f"results/{self.model_name}/text/Results_param_hunt_{self.model_name}_{bg}.txt", 'a') as r_file:
                r_file.write(
                    f"Trial {trial.number} | F1: {trial.value:.4f} |"
                    f"LR {trial.params['lr']:.6f} | Dropout: {trial.params['dropout']:.4f}\n"
                )

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials = 25, callbacks=[save_trial]) #25 combinations
        if self.emissions:
            emission.stop()

        #Here just write the best param
        with open(f"results/{self.model_name}/text/Results_param_hunt_{self.model_name}_{bg}.txt", 'a') as r_file:
            r_file.write(
                f"\n Best LR: {study.best_params['lr']:.6f} |"
                f"Dropout: {study.best_params['dropout']:.4f}| "
                f"F1: {study.best_value:.4f}\n"
                f"Total Emissions: {emission if self.emissions else {}} \n"
            )         

        
