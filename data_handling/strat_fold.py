"""This modules deals with the stratified K-fold cross evaluation."""
from collections import defaultdict
import numpy as np 

class StratifiedFold:
    """
    This module deals with handling and splitting the dataset into K number of folds.

    Attributes:
        k (int) = The number of splits for the k-fold cross evaluation.
        fold (list) = The fold indices.

    Methods:
        stratifier(df: pd.DataFrame, column:list[str]) -> None
            Deals with stratifying and splitting the dataset into k-folds.

        __iter__() -> list, list
            Allows the user to iterate over the folds.
    """
    def __init__(self, k=10):
        self.k = k
        self.folds = []

    def stratifier(self, df: pd.DataFrame, column: list[str]) -> None:
        """
        This method deals with both stratifying each fold and diving the dataset into folds.

        Arguments:
            df (pd.DataFrame) = A dataframe with the test to be split.
            column (list(str)) = The list of the relevant columns from which information should be extracted.
        """
        label2id = defaultdict(lambda: len(label2id))

        col_lab = df[column].values

        labels = np.unique(col_lab)

        #add to labels
        [label2id[label] for label in labels]
        
        col_val = np.array([label2id[label] for label in col_lab])


        n_classes = len(np.unique(col_val))
        #n_samples = len(col_val)

        class_counts = np.bincount(col_val)
        n_samples_per_class = class_counts // self.k
        remainder = class_counts % self.k
        
        
        
        for fold in range(self.k):
            val_indices = []
            train_indices = []

            for class_id in range(n_classes):
                mask = (col_val == class_id)
                class_idx = np.where(mask)[0]
                
                extra_before = min(fold, remainder[class_id])
                start = fold * n_samples_per_class[class_id] + extra_before
                end = start + n_samples_per_class[class_id]

                
                if class_id < len(remainder) and fold < remainder[class_id]:
                    end += 1
                val_indices.extend(class_idx[start:end])

                train_indices.extend(np.concatenate([class_idx[:start], class_idx[end:]]))
                
            self.folds.append((np.array(train_indices), np.array(val_indices)))

            

    def __iter__(self) -> Tuple[list, list]:
        """
        This methods enables iteration through the training folds and validation folds.

        Returns|Yields:
            train_ids (list) = The indices for the training fold.
            val_ids (list) = The indices for the validation fold.
        """
        for train_ids, val_ids in self.folds:
            yield train_ids, val_ids
