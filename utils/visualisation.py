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
        def __init__(self, model_stats, model_preds, labels, label_names, pairwise_result=None, save_path="results/"):

            self.model_stats = model_stats
            self.model_preds = model_preds
            self.labels = labels
            self.label_names = label_names
            self.pairwise_result= pairwise_result or []
            self.model_names = list(model_stats.keys())
            self.save_path = save_path

        


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
            
            if self.save_path:
                plt.savefig(f'{self.save_path}/{model_name}/images/conf_matrix.png', dpi=300, bbox_inches='tight')

            plt.show()
        
        def plot_f1_distribution(self, alpha=0.005):
             
            fig, ax = plt.subplots(figsize=(9, 5))
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
            if self.save_path:
                plt.savefig(f'{self.save_path}/{model_name}/images/f1_distribution.png', dpi=300, bbox_inches='tight')
            plt.show()
        

        def plot_pairwise(self):
             
            n = len(self.pairwise_result) 
            fig, ax = plt.subplots(figsize=(8, n * 1.2 + 1.5))

            for i, r in enumerate(self.pairwise_result):
                color = 'green' if r['significant'] else 'gray'
                #label = f"{r['model_a']} vs {r['model_b']}"
                
                ax.barh(i, r['mean_diff'], color=color, alpha=0.7, height=0.3)
                ax.text(
                    0.02, i,
                    #r['mean_diff'], i,
                    f" p={r['p_corrected']:.4f}{'*' if r['significant'] else ''} Δ={r['mean_diff']:.3f}",
                    va='center', ha='left', transform=ax.get_yaxis_transform()
                )  

            ax.axvline(0, color='black', linewidth=1.5, linestyle='--')
            ax.set_yticks(range(n))
            ax.set_yticklabels([f"{r['model_a']} vs {r['model_b']}" for r in self.pairwise_result])
            ax.set_ylim(-0.5, n - 0.5)
            ax.set_xlabel("Mean F1 difference (A - B)\n← B better | A better →)")
            ax.set_title("Pairwise comparisom(BH-corrected)\ngreen = significant")
            plt.tight_layout()
            if self.save_path:
                plt.savefig(f'{self.save_path}/pairwise_comparison.png', dpi=300, bbox_inches='tight')
            plt.show()      



