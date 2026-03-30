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

class SigTest:
    def __init__(self, df, evaluate, models, label, n_runs=10000):
        self.test = df 
        self.models = models #a list of names
        self.n_runs = n_runs #10.000s
        self.evaluate = evaluate
        self.n_classes = len(set(self.test[label]))
    
    def chance_test(self):
        print(self.models)        
        pm = PathManager()
        
        ModelResult = namedtuple('ModelResult', ['model_name', 'preds'])
        model_preds = []
        labels = None
        for model_name in self.models:
            
            model = pm.get_model(model_name)
            print("We are printing model: ", model)
#            model = torch.load(
            all_preds, all_labels, f1, pre, rec, acc = self.evaluate(val_data=self.test, model=str(model), boot=True)
            model_preds.append(
                ModelResult(
                    model_name=f'{model_name}',
                      preds=all_preds
                      ))
            labels = all_labels

        #bootstrap set (n_runs, test_size), (10.000, test_size)   
        boot_set = np.random.choice(len(labels), size=(self.n_runs, len(labels)), replace=True)
        
        boot_labels = labels[boot_set]

        boot_scores = {model_name.model_name: [] for model_name in model_preds}
        
        for model_name in model_preds:
            boot_preds = model_name.preds[boot_set]

            for i in range(self.n_runs):
                prec, rec, f1, _ = precision_recall_fscore_support(boot_labels[i], boot_preds[i], average='macro', zero_division=0)
                
                boot_scores[model_name.model_name].append((prec, rec, f1))
        stats = self.bootstrap_stats(boot_scores)
        print(stats)
        return stats, boot_scores


    def bootstrap_stats(self, boot_scores, alpha=0.005):

        ChanceResult = namedtuple('ChanceResult', ['model_name', 'mean_f1', 'ci_lower', 'ci_upper', 'p_value'])

        model_stats = {}
        for model, boot_scores in boot_scores.items():
            #f1 index f1

            f1s = np.array([s[2] for s in boot_scores])

            mean_f1 = np.mean(f1s)

            #Confidence interval: the rangge within the F1 likely falls

            ci_lower = np.percentile(f1s, 100 * alpha / 2)
            ci_upper = np.percentile(f1s, 100 * (1-alpha / 2))

            p_value = self.compute_chance_pvalue(f1s)

            model_stats[model] = {
                'f1s' : f1s,
                'result': ChanceResult(model, mean_f1, ci_lower, ci_upper, p_value)
            }
        
        return model_stats
    
    def compute_chance_pvalue(self, f1s):

        chance_level = 1 /self.n_classes

        #f1s = np.array([s[2] for s in boot_scores[model_name]])

        p_value = np.mean(f1s <= chance_level)

        return p_value

    def pairwise_test(self, boot_scores):

        model_stats = self.bootstrap_stats(boot_scores)

        PairwiseResult = namedtuple('PairwiseResult', [
            'model_a', 'model_b', 'p_value', 'mean_diff'
        ])
        pairwise_results = []

        for model_a, model_b in combinations(model_stats.keys(), 2):
            f1s_a = model_stats[model_a]['f1s']
            f1s_b = model_stats[model_b]['f1s']

            #difference per bootstrap run
            diffs = f1s_a - f1s_b
            mean_diff = np.mean(diffs)

            #p-value
            if mean_diff >= 0:
                p_value = np.mean(diffs <= 0)
            else:
                p_value = np.mean(diffs >= 0)
            
            pairwise_results.append(
                PairwiseResult(model_a, model_b, p_value, mean_diff)            
            )
        print(pairwise_results)



    

    

                            
                            

                            
                        

                        
            
