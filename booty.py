import pandas as pd
from tqdm import tqdm
import numpy as np
import torch
from scipy import stats
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

class SigTest:
    def __init__(self, df, models, label, n_samples, n_epochs, alpha=0.005):
        self.test = df
        self.models = models
        self.n_samples = n_samples 
        self.n_epochs = n_epochs
    
    def get_bootstrap_loader(self):
        indices = np.random.choice(self.num_samples, size=self.n_samples, replace=True)

        dataloader = DataLoader(Subset(self.test, indices))

        return dataloader

    def bootstrap_chance(self):
        for i in tqdm(range(10), description="Bootstraping against chance"):

    def bootstrap(self, models):

        
        results = []

        for i in tqdm(range(10)):

            dataloader = self.get_bootstrap_loader()

            all_preds = []
            true_lables = []
            prev_batch_size_p = None

            for s in dataloader: 

                p_tokens, p_features, p_labels, p_conv_ids = pros_batch
                lex_tokens, lex_features, lex_labels, lex_conv_ids = lex_batch

                with torch.no_grad():
                    
                    current_batch_size = p_tokens.size(0)
                    
                    if current_batch_size != prev_batch_size_p:
                        hidden_pros = pros_model.init_hidden(current_batch_size)
                        hidden_lex = lexical_model.init_hidden(current_batch_size)
                    #      prev_conv_id = conv_ids[0].item()
                    prev_batch_size_p = current_batch_size
                                
                    pros_preds, hidden1 = pros_model(p_tokens, hidden_pros, p_features) 
                    lex_preds, hidden2 = lexical_model(lex_tokens, hidden_lex, lex_features)

                    labels = p_labels.float()

                    #pros_preds = pros_preds.squeeze(-1)
                    #lex_preds = lex_preds.squeeze(-1)

                    pros_preds = (pros_preds[:, 15] > 0.5).float().cpu()
                    lex_preds = (lex_preds[:, 15] > 0.5).float().cpu()
                    labels= labels[:, 15].cpu()

                    all_pros_preds.append(pros_preds)
                    all_lex_preds.append(lex_preds)
                    true_lables.append(labels)
            #ALL Batches
            pros_preds_fb = torch.cat(all_pros_preds).numpy()
            lex_preds_fb = torch.cat(all_lex_preds).numpy()
            all_labels = torch.cat(true_lables).numpy()

            # Calculate metrics for prosodic model
            pros_prec, pros_rec, pros_f1, _ = precision_recall_fscore_support(
                all_labels, pros_preds_fb, average='binary', zero_division=0
            )
            pros_acc = accuracy_score(all_labels, pros_preds_fb)

            pros_results.append({
                'iteration': i,
                'accuracy': pros_acc,
                'precision': pros_prec,
                'recall': pros_rec,
                'f1': pros_f1
            })


            # Calculate metrics for lexical model
            lex_prec, lex_rec, lex_f1, _ = precision_recall_fscore_support(
                all_labels, lex_preds_fb, average='binary', zero_division=0
            )
            lex_acc = accuracy_score(all_labels, lex_preds_fb)

            lex_results.append({
                'iteration': i,
                'accuracy': lex_acc,
                'precision': lex_prec,
                'recall': lex_rec,
                'f1': lex_f1
            })
        
        pros_df = pd.DataFrame(pros_results)
        lex_df = pd.DataFrame(lex_results)

        self.plot_confusion_matrix(
            all_labels, pros_preds_fb,
            model_name="Prosodic Model",
            save_path=f'pros_confusion_matrix_iteration.png'
        )
        self.plot_confusion_matrix(
            all_labels, lex_preds_fb,
            model_name="Lexical Model",
            save_path=f'lex_confusion_matrix_iteration.png'
        )
        
        return pros_df, lex_df

    def t_tester(self, pros_df, lex_df):

        diff_f1 = pros_df['f1'] - lex_df['f1']
        diff_ci_lower = diff_f1.quantile(0.025)
        diff_ci_upper = diff_f1.quantile(0.975)

        t_stat, p_value = stats.ttest_rel(
                        pros_df['f1'],
                        lex_df['f1']
                        #alternative='greater'
                        )
        print(f"Model Prosodic: Mean F1 = {pros_df['f1'].mean():.4f}")
        print(f"Modal Lexical: Mean F1 = {lex_df['f1'].mean():.4f}")
        print(f"Mean difference (Pros - Lex) = {(pros_df['f1'] - lex_df['f1']).mean():.4f}")
        print(f"95% CI for difference: [{diff_ci_lower:.4f}, {diff_ci_upper:.4f}]")
        print(f"p-value (two-tailed) = {p_value:.20f}")

        if p_value < 0.05:
            if (pros_df['f1'] - lex_df['f1']).mean() > 0:
                print("Prosodic model significantly outperforms Lexical model (p < 0.05)")
            else:
                print("Lexical model significantly outperforms Prosodic model (p < 0.05)")
        else:
            print("No statistically significant difference between models")


        print(f"\nModel Prosodic std dev: {pros_df['f1'].std():.4f}")
        print(f"Model Lexical std dev: {lex_df['f1'].std():.4f}")

        # Visualize
        

        plt.figure(figsize=(6, 5))
        #plt.subplot(1, 2, 1)
        #plt.hist(pros_df['f1'], bins=30, alpha=0.7, label='Model Prosodic')
        #plt.hist(lex_df['f1'], bins=30, alpha=0.7, label='Model Lexical')
        #plt.xlabel('F1 Score')
        #plt.ylabel('Frequency')
        #plt.legend()
        #plt.title('Bootstrap Distribution')

        #plt.subplot(1, 2, 2)
        plt.boxplot([pros_df['f1'], lex_df['f1']], labels=['Model Prosodic', 'Model Lexical'])
        plt.ylabel('F1 Score')
        plt.title('Comparison')
        plt.tight_layout()
        plt.savefig('model_comparison.png')

    def plot_confusion_matrix(self, y_true, y_pred, model_name, save_path=None):
    
        cm = confusion_matrix(y_true, y_pred)
        print(f"Confusion Matrix for {model_name}:")
        print(cm)

        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Predicted non jump-in', 'predicted-jump in'],
                    yticklabels=['Actual non jump-in', 'Actual jump-in'])
        plt.title(f'{model_name} Confusion Matrix')
        if save_path:
            plt.savefig(save_path)
        plt.show()
        plt.close()

                            
                            

                            
                        

                        
            
