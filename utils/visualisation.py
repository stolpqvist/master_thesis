import pandas as pd
import numpy as np
import torch
from scipy import stats
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from sig_test import SigTest

class Visual:
        def __init__(self, model_stats, model_preds, labels, label_names, pairwise_result=None):

            self.model_stats = model_stats
            self.model_preds = model_preds
            self.labels = labels
            self.label_names = label_names
            self.pairwise_result= pairwise_result or []
            self.model_names = list(model_stats.keys)

        


        def plot_confusion_matrix(self):

            n = len(self.model_names)
            fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
            if n==1:
                axes = [axes]
            
            for ax, model_name in zip(axes, self.model_names):
                preds = self.model_preds[model_name]
                cm = confusion_matrix(self.labels, preds)
                cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

                sns.heatmap(
                    cm_norm,
                    annot=cm,
                    fmt='d',
                    cmap='Blues',
                    xticklabels=self.label_names,
                    yticklabels=self.label_names,
                    vmin=0, vmax=1,

                    ax=ax
                )

                result = self.model_stats[model_name]['result']
                ax.set_title(f"{model_name}\nF1={result.mean_f1:.3f} p={result.p_value:.4f}")
                ax.set_xlabel('Predicted')
                ax.set_ylabel('True')

            plt.tight_layout()
            plt.show()
        
        def plot_f1_distribution(self, alpha=0.005):
             
            fig, ax = plt.subplot(figsize=(9, 5))
            colors = plt.cm.tab10.colors

            for i, model_name in enumerate(self.model_names):
                f1s = self.model_stats[model_name]['f1s']
                result = self.model_stats[model_name]['result']
                color = colors[i]

                ax.hist(f1s, bins=60, alpha=0.4, color=color, label=model_name)

                ax.axvline(result.mean_f1, color=color, linewidth=2, linestyle='-')
                ax.axvline(result.ci_lower, color=color, linewidth=1, linestyle='--')
                ax.axvline(result.ci_upper, color=color, linewidth=1, linestyle='--')
            
            chance = 1/ len(self.label_names)
            ax.axvline(chance, color='red', linewidth=1.5, linestyle=':', label=f"Chance ({chance:.2f})")

            ax.set_xlabel('Macro F1')
            ax.set_ylabel('Bootstrap ferquency')
            ax.set_title('Bootstrap F1 distributions')
            ax.legend()
            plt.tight_layout()
            plt.show()



