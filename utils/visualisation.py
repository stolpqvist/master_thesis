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
        def __init__(self, bg, model_stats, model_preds, labels, label_names, pairwise_result=None, save_path="results"):

            self.bg = bg
            self.model_stats = model_stats
            self.model_preds = model_preds
            self.labels = labels
            self.label_names = label_names
            self.pairwise_result= pairwise_result or []
            self.model_names = list(model_stats.keys())
            self.save_path = save_path



        def plot_confusion_matrix(self):
            import os

            for model_name in self.model_names:
                save_file = f"{self.save_path}/{model_name}/images/conf_matrix_{self.bg}.png"

                print(model_name, save_file)
            
                if self.save_path and os.path.exists(save_file):
                    continue

                #n = len(self.model_names)
                fig, ax = plt.subplots(figsize=(6, 5))
                
                #if n==1:
                #    axes = [axes]
                
                #for ax, model_name in zip(axes, self.model_names):
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
                    os.makedirs(os.path.dirname(save_file), exist_ok=True)
                    plt.savefig(save_file, dpi=300, bbox_inches='tight')
                    print(f"[saved] {model_name}: confusion matrix saved. ")
                    #plt.savefig(f'{self.save_path}/{model_name}/images/conf_matrix_{self.bg}.png', dpi=300, bbox_inches='tight')

                plt.show()
                plt.close(fig)
        
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
            ax.set_title(f"{self.bg} Bootstrap F1 distributions")
            ax.legend()
            plt.tight_layout()
            if self.save_path:
                plt.savefig(f'{self.save_path}/boot/f1_distribution_{self.model_names[0]}_{self.model_names[1]}_{self.bg}.png', dpi=300, bbox_inches='tight')
            plt.show()
        
        def box_plot_f1(self):
            fig, ax = plt.subplots(figsize=(8, 5))
            colors = plt.cm.tab10.colors

            data = [self.model_stats[m]['f1s'] for m in self.model_names]

            bp = ax.boxplot(
                data,
                patch_artist=True,
                notch=True,
                vert=True,
                label=self.model_names
            )


            #color each box individually
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.6)

            for element in ['whiskers', 'caps', 'medians', 'fliers']:
                for item, color in zip(
                    bp[element],
                    [c for c in colors for _ in range(2 if element in ('whiskers', 'caps') else 1)]

                ):
                    item.set_color(color)
            

            chance = 1 / len(self.label_names)
            ax.axhline(chance, color='red', linewidth=1.5, linestyle=':', label=f"Chance ({chance:.2f})")

            ax.set_xlabel('Model')
            ax.set_ylabel('Macro F1')
            ax.set_title(f'{self.bg} Bootstrap F1 Distribution (Box Plot)')
            ax.legend()
            plt.tight_layout()

            if self.save_path:
                plt.savefig(f'{self.save_path}/boot/f1_boxplot__{self.model_names[0]}_{self.model_names[1]}_{self.bg}.png', dpi=300, bbox_inches='tight')
            plt.show()

        #def plot_pairwise(self):
             
            # n = len(self.pairwise_result) 
            # fig, ax = plt.subplots(figsize=(8, n * 1.2 + 1.5))

            # for i, r in enumerate(self.pairwise_result):
            #     color = 'green' if r['significant'] else 'gray'
            #     #label = f"{r['model_a']} vs {r['model_b']}"
                
            #     ax.barh(i, r['mean_diff'], color=color, alpha=0.7, height=0.3)
            #     ax.text(
            #         0.02, i,
            #         #r['mean_diff'], i,
            #         f" p={r['p_corrected']:.4f}{'*' if r['significant'] else ''} Δ={r['mean_diff']:.3f}",
            #         va='center', ha='left', transform=ax.get_yaxis_transform()
            #     )  

            # ax.axvline(0, color='black', linewidth=1.5, linestyle='--')
            # ax.set_yticks(range(n))
            # ax.set_yticklabels([f"{r['model_a']} vs {r['model_b']}" for r in self.pairwise_result])
            # ax.set_ylim(-0.5, n - 0.5)
            # ax.set_xlabel("Mean F1 difference (A - B)\n← B better | A better →)")
            # ax.set_title("Pairwise comparisom(BH-corrected)\ngreen = significant")
            # plt.tight_layout()
            # if self.save_path:
            #     plt.savefig(f'{self.save_path}/pairwise_comparison.png', dpi=300, bbox_inches='tight')
            # plt.show()      



