import argparse
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold, train_test_split


from preprocessing.pre_roberta import DataProcessor
from model.roberta import CustomXLMRoberta
from train.train import ModelTrain




def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('-bg', type=str, default='NT')
    parser.add_argument('-sp', action='store_true', default=False) #Split datatsets into leave one out validation
    parser.add_argument('-md', type=str, default= 'roberta') #model
    parser.add_argument('-dr', type=float, default=0.5)
    parser.add_argument('-lr', type=float, default=0.0001)
    parser.add_argument('-e', type=int, default=30) #epochs
    parser.add_argument('-tr', action='store_true', default=False)
    parser.add_argument('-test_size', type=float, default=0.1)       # Test size
    args = parser.parse_args()

    if args.bg:
        bg = args.bg
    
    
    if args.tr:
        file = f"../datasets/{bg}/{bg}_dataset.csv"
        df = pd.read_csv(file)

        label_cl = 'TilldeladBeredningsgruppKortNamn'

        df_trainval, df_test = train_test_split(
            df,
            test_size=args.test_size,
            random_state=42,
            stratify=df[label_cl]
        )

        df_trainval = df_trainval.reset_index(drop=True)
        df_test     = df_test.reset_index(drop=True)

        print(f"Dataset sizes  →  train+val: {len(df_trainval)}  |  test: {len(df_test)}")

        trainval_tmp = f"/tmp/{bg}_trainval.csv"
        df_trainval.to_csv(trainval_tmp, index=False)
        
        from collections import defaultdict

        label2id = defaultdict()
        id2label = {}

        for label in df_trainval[label_cl]:
            if label not in label2id:
                lid = len(label2id)
                label2id[label] = lid
                id2label[lid] = label

        num_classes = len(label2id)
        print(f"Number of classes: {num_classes}")

        # Then create dp_full separately for the trainer
        dp_full = DataProcessor(trainval_tmp)
        
        print(f"Number of classes: {num_classes}")

        print("\n Final training on full train+val set")
        test_tmp = f"/tmp/{bg}_test.csv"

    
    if args.sp:
        kf = KFold(n_splits=10, shuffle=False)
        


        #indices = df_trainval.index.to_numpy()
        
        fold_results = []

        print(f"\nStarting KFold cross-validation ({len(indices)} folds)...")
        labels = df['TilldeladBeredningsgruppKortNamn']
        data = df[['Beskrivning', 'BeskrivningEng', 'AnsökanTitelEng', 'AnsökanTitel']]
        for fold, (train_idx, val_idx) in enumerate(kf.split(data, labels)):
            print(f"\n── KFold fold {fold + 1}/{len(indices)} ──")
            
            df_fold_train = df_trainval.iloc[train_idx].reset_index(drop=True)
            df_fold_val   = df_trainval.iloc[val_idx].reset_index(drop=True)

        # Write fold splits to temp CSVs to keep DataProcessor interface intact
            fold_train_tmp = f"/tmp/{bg}_fold_train.csv"
            fold_val_tmp   = f"/tmp/{bg}_fold_val.csv"
            df_fold_train.to_csv(fold_train_tmp, index=False)
            df_fold_val.to_csv(fold_val_tmp,     index=False)

            dp_train = DataProcessor(fold_train_tmp)
            dp_val   = DataProcessor(fold_val_tmp)

            model = CustomXLMRoberta(
                    num_classes=num_classes,
                    hidden_dropout=args.dr
                )

            trainer = ModelTrain(
                model=model,
                train_data=dp_train,
                val_data=dp_val,
                label2id=label2id,
                id2label=id2label,
                lr=args.lr,
                epochs=args.e
                )

            fold_val_score = trainer.train()
            fold_results.append(fold_val_score)
            print(f"Fold {fold + 1} validation score: {fold_val_score:.4f}")

        avg = sum(fold_results) / len(fold_results)
        print(f"\nLOO mean validation score across {len(fold_results)} folds: {avg:.4f}")

    print("\n── Final training on full train+val set ──")

    test_tmp = f"/tmp/{bg}_test.csv"
    df_test.to_csv(test_tmp, index=False)

    # dp_full already holds the full train+val data, reuse it
    dp_test = DataProcessor(test_tmp)

    final_model = CustomXLMRoberta(
        num_classes=num_classes,
        hidden_dropout=args.dr
    )

    final_trainer = ModelTrain(
        model=final_model,
        train_data=dp_full,
        val_data=dp_test,
        label2id=label2id,
        id2label=id2label,
        lr=args.lr,
        epochs=args.e
    )

    test_score = final_trainer.train()
    print(f"\nFinal test set score: {test_score:.4f}")

if __name__ == "__main__":
    main()




