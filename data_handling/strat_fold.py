from collections import defaultdict
from numpy import np 

class StratifiedFold:

    def __init__(self, dataset, k=10, label2id=None):
        self.dataset = dataset
        self.k = k
        self.label2id = label2id
        self.fold = []

    def stratifier(self, df, column='TilldeladBeredningsgruppKortNamn'):
        col_lab = df[column].values
        col_val = np.array([self.label2id[label] for label in col_lab])


        n_classes = len(np.unique(col_val))
        n_samples = len(col_val)

        class_counts = np.bincount(col_val)
        n_samples_per_class = class_counts // self.k
        remainder = class_counts % self.k
        
        
        
        for fold in range(self.k):
            val_indices = []
            train_indices = []

            for class_id in class_indices:
                indices = class_indices[class_id]

                n_samples = len(indices)

                n_samples_per_fold = n_samples // self.k
                remainder = n_samples % self.k
                start = fold * n_samples_per_fold
                end = start + n_samples_per_fold

                if fold < remainder:
                    end += 1
                val_indices.extend(indices[start:end])

                train_indices.extend(indices[:start] + indices[end:])
            self.folds.append((np.array(train_indices), np.array(val_indices))

            



    def __iter__(self):
        for train_idx, val_idx in self.folds:
            yield np.array(train_indices), np.array(val_indices)
