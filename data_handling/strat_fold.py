from collections import defaultdict
from numpy import np 

class StratifiedFold:

    def __init__(self, dataset, k=10, label2id=None):
        self.dataset = dataset
        self.k = k
        self.label2id = label2id
        self.folds = []

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

            for class_id in range(n_classes):
                mask = (y == class_id)
                class_idx = np.where(class_mask)[0]

                start = fold * n_samples_per_class[class_id]
                end = start + n_samples_per_class[class_id]

                
                if class_id < len(remainder) and fold < remainder[class_id]:
                    end += 1
                val_indices.extend(indices[start:end])

                train_indices.extend(indices[:start] + indices[end:])
            self.folds.append((np.array(train_indices), np.array(val_indices))

            



    def __iter__(self):
        for train_idx, val_idx in self.folds:
            yield train_idx, val_indices
