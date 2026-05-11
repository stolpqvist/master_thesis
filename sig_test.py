"""
This module handles the significance testing, it offers both testing against chance and testing models against eachother with the
subsequent Bonferroni correction to account for increasing rates of false positives.
"""
import pandas as pd
from tqdm import tqdm
import numpy as np
import torch
from scipy import stats
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix
from collections import namedtuple
from itertools import combinations
from utils.path_manager import PathManager
from statsmodels.stats.multitest import multipletests

class SigTest:
    def __init__(self, df, evaluate, models, bg, label, n_runs=10000):
    """
    This class handles the significance testing of the model, it can either test one or multiple models against chance, or against
    eachother. It is also able to compare the models against eachother with a pairwise test with a subsequent Bonferroni correction.

    Attributes:
        test (pd.DataFrame) = A pandas dataframe with the file for the test set.
        models (list(str)) = A list of the models to be used.
        n_runs (int) = The amount of times to bootstrap.
        evaluate (ExperimentOrganiser Object) = The method used for evaluation.
        bg (str) = The dataset group that the model will be tested on.
        n_classes (int) = The total amount of classes.
        labels = The true labels to be compared against.
        model_preds = The predictions of a model.

    """
        self.test = df 
        self.models = models #a list of names
        self.n_runs = n_runs #10.000s
        self.evaluate = evaluate
        self.bg = bg
        self.n_classes = len(set(self.test[label]))
        self.labels = None
        self.model_preds = None
    
    def chance_test(self) -> list[float]:
        """
        This method handles testing the models against chance.

        Returns:
            stats (list(float)) = The bootstrapping scores of the models.

        """

        pm = PathManager()
        
        ModelResult = namedtuple('ModelResult', ['model_name', 'preds'])
        model_preds = []
        labels = None

        for model_name in self.models:
            
            model = pm.get_model(model_name, self.bg)
            print("We are printing model: ", model)
#            model = torch.load(
            all_preds, all_labels, f1, pre, rec, acc = self.evaluate(val_data=self.test, model=str(model), boot=True)
            model_preds.append(
                ModelResult(
                    model_name=f'{model_name}',
                      preds=np.array(all_preds)
                      ))
            
            #only once, bc they are shuffled othersie
            if labels is None:
                labels = all_labels
        
        #Original preds for conf matrices:
        self.model_preds = {m.model_name: m.preds for m in model_preds}
        self.labels = labels

        #bootstrap set (n_runs, test_size), (10.000, test_size)   
        boot_set = np.random.choice(len(labels), size=(self.n_runs, len(labels)), replace=True)
        
        boot_labels = labels[boot_set]

        boot_scores = {model_name.model_name: [] for model_name in model_preds}
        
        for model_name in model_preds:
            boot_preds = model_name.preds[boot_set]

            for i in range(self.n_runs):
                prec, rec, f1, _ = precision_recall_fscore_support(boot_labels[i], boot_preds[i], average='macro', zero_division=0)
                
                boot_scores[model_name.model_name].append((prec, rec, f1))

        #returen stats
        stats = self.bootstrap_stats(boot_scores, self.model_preds, self.labels)
        
        return stats


    def bootstrap_stats(self, boot_scores, preds_dict, labels, alpha=0.05):
        """
        This method handles the calculation of the p-value and confidence intervals.

        Arguments:
            boot_scores (list(float)) = list of boot scores that the models have already achieved.
            preds_dict (dict) = A dictionary containing the predictions.
            labels (list) = A list of all the actual labels.
            alpha (float) = The alpha level to be used for this test
        
        Returns:
            model_stats (dict) = A dictionary containing the model, mean_f1, confidence intervals, and p-value.
        """


        ChanceResult = namedtuple('ChanceResult', ['model_name', 'mean_f1', 'ci_lower', 'ci_upper', 'p_value'])

        model_stats = {}
        for model, scores in boot_scores.items():
            #f1 index f1

            f1s = np.array([s[2] for s in scores])

            mean_f1 = np.mean(f1s)

            #Confidence interval: the rangge within the F1 likely falls

            ci_lower = np.percentile(f1s, 100 * (alpha / 2))
            ci_upper = np.percentile(f1s, 100 * (1 - alpha / 2))

            p_value, f1_obs = self.compute_chance_pvalue(preds=preds_dict[model], labels=labels)

            model_stats[model] = {
                'f1s' : f1s,
                'f1_obs': f1_obs,
                'result': ChanceResult(model, mean_f1, ci_lower, ci_upper, p_value)
            }
        
        return model_stats
    
    def compute_chance_pvalue(self, preds, labels, n_perm=1000):
        """
        This method handles the actual computation of the p-value.

        Arguments:
            preds = The predictions of a model.
            labels = The actual labels to compare against.
            n_perm (int) = The actual number of permutations to do.

        Returns:
            p_value (float): The p_value for the model's performance.
            f1_obs (float) : The F1-score of a model.
        """

        #observed score:
        _, _, f1_obs, _ = precision_recall_fscore_support(labels, preds, average='macro', zero_division=0)

        #null scores:
        f1_null = []
        for _ in range(n_perm):
            perm_labels = np.random.permutation(labels)
            _, _, f1_perm, _ = precision_recall_fscore_support(perm_labels, preds, average='macro', zero_division=0)
            f1_null.append(f1_perm)

        #chance_level = 1 /self.n_classes

        #f1s = np.array([s[2] for s in boot_scores[model_name]])

        f1_null = np.array(f1_null)

        p_value = np.mean(f1_null >= f1_obs)

        return p_value, f1_obs

    def pairwise_test(self):
        """
        This method deals with testing the models against eachother.

        Returns:
            pairwise_results (list(dict)) = The results of models' performance being compared against eachother
            model_stats (dict) = A dictionary with the performance of the models.

        """
        model_stats = self.chance_test()

        #model_stats = self.bootstrap_stats(boot_scores)

        #PairwiseResult = namedtuple('PairwiseResult', [
        #    'model_a', 'model_b', 'p_value', 'mean_diff'
        #])
        pairwise_results = []

        for model_a, model_b in combinations(model_stats.keys(), 2):
            f1s_a = model_stats[model_a]['f1s']
            f1s_b = model_stats[model_b]['f1s']

            observed_diff = model_stats[model_a]['f1_obs'] - model_stats[model_b]['f1_obs']

            #difference per bootstrap run
            diffs = f1s_a - f1s_b
            diffs_centered = diffs - np.mean(diffs)

           

            #two tailed t test
            p_value = np.mean(np.abs(diffs_centered) >= np.abs(observed_diff))

            # Determine which model is better (if significant)
            better_model = None
            if p_value < 0.05:
                better_model = model_a if observed_diff > 0 else model_b
            
            pairwise_results.append({
                'model_a': model_a,
                'model_b': model_b, 
                'p_value': p_value, 
                'mean_diff': observed_diff,
                'better_model': better_model
            })
        

        self.p_value_correction(pairwise_results)
        print(pairwise_results)
        return pairwise_results, model_stats
    
    def p_value_correction(self, pairwise_results):
        """
        Bonferroni-based for the pairwise test.

        Arguments:
            pairwise_results (list) : The pairwise results when comparing model A to model B.

        Returns:
            pairwise_results (list) = The pairwise results with the added Bonferroni correction.
        """

        p_values = [r['p_value'] for r in pairwise_results]
        reject, p_corrected, _, _ = multipletests(p_values, alpha=0.05, method='bonferroni')

        for r, p_corr, sig in zip(pairwise_results, p_corrected, reject):
            r['p_corrected'] = p_corr
            r['significant'] = sig
        
        return pairwise_results




    

    

                            
                            

                            
                        

                        
            
