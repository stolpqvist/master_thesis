from sklearn.model_selection import train_test_split
import os
import numpy as np
import pandas as pd
from imblearn.over_sampling import RandomOverSampler

def Preprocess():

    data = input("HS/NT/ALL?").upper()

    print("ProjektSammanfattningEng/Nyckelord/AnsökanTitelEng?")
    column = input("PS/N/A: ")
    if column == "PS":
        column = "ProjektSammanfattningEng"

    if column == "N":
        column = "Nyckelord"

    if column == "A":
        column = "AnsökanTitelEng"

    if data == "":
        data = input("HS/NT?: ")
    if data == "ALL":
        column = ["ProjektSammanfattningEng", "Nyckelord", "AnsökanTitelEng"]
    if column == "ALL":
        column = ["ProjektSammanfattningEng", "Nyckelord", "AnsökanTitelEng"]
    
    print("balanced or imbalanced dataset")
    balance = input("B/U?:" ).upper()
    if balance == 'B':
        balance = 'balanced'
    if balance == 'U':
        balance = 'unbalanced'


    data = data.upper()
    df = pd.read_csv(f'Data/{data}_data.csv')

    # %%
    df.head()

    # %%
    df['TilldeladBeredningsgruppKortNamn'].value_counts()

    # %%
    possible_labels = df.TilldeladBeredningsgruppKortNamn.unique()

    label_dict = {}
    for index, possible_label in enumerate(possible_labels):
        label_dict[possible_label] = index
    label_dict

    # %%
    df['label'] = df.TilldeladBeredningsgruppKortNamn.replace(label_dict)

    # %%
    df.head()
    
    if type(column) == list:
       df['ALL'] = df['Combined_Column'] = df['ProjektSammanfattningEng'] + ' ' + df['Nyckelord'] + ' ' + df['AnsökanTitelEng']
       column = 'ALL'
    
    if not os.path.exists(f"./experiment_data"):
        os.makedirs(f"./experiment_data")

    if not os.path.exists(f"./experiment_data/{data}"):
        os.makedirs(f"./experiment_data/{data}")

    if not os.path.exists(f"./experiment_data/{data}/{column}"):
        os.makedirs(f"./experiment_data/{data}/{column}")

    if not os.path.exists(f"./experiment_data/{data}/{column}/{balance}"):
        os.makedirs(f"./experiment_data/{data}/{column}/{balance}")
        os.mkdir(f"./experiment_data/{data}/{column}/{balance}/training_data")
        os.mkdir(f"./experiment_data/{data}/{column}/{balance}/validation_data")
        os.mkdir(f"./experiment_data/{data}/{column}/{balance}/test_data")
        os.mkdir(f"./experiment_data/{data}/{column}/{balance}/models")

    data_paths = [f'./experiment_data/{data}training_data/X_train.csv',
                   f'./experiment_data/{data}/training_data/y_train.csv',
                     f'./experiment_data/{data}/validation_data/X_val.csv',
                       f'./experiment_data/{data}/validation_data/y_val.csv',
                         f'./experiment_data/{data}/test_data/X_test.csv',
                           f'./experiment_data/{data}/test_data/y_test.csv'
                           ]
    ros = RandomOverSampler(sampling_strategy='not majority')


    # Removes NULL or NaN rows
    df.dropna(subset=['TilldeladBeredningsgruppKortNamn'], inplace=True)
    label_counts = df['TilldeladBeredningsgruppKortNamn'].value_counts()
    label_counts_str = label_counts.to_string()

    # Encode the string as bytes
    label_counts_bytes = label_counts_str.encode('utf-8')

    # Write the bytes to the file
    with open('label_counts.txt', 'wb') as label_file:
        label_file.write(label_counts_bytes)



    #train_test_split(Corpus['ProjektSammanfattningEng'], Corpus['TilldeladBeredningsgruppKortNamn'], random_state = 0)
    X_train, X_val_test, y_train, y_val_test = train_test_split(df[f'{column}'], 
                                                    df['TilldeladBeredningsgruppKortNamn'], 
                                                    test_size=0.2, 
                                                    random_state=42, 
                                                    stratify=df['TilldeladBeredningsgruppKortNamn']
                                                    )
    X_train = np.array(X_train)
    #print(X_train)
    if balance == 'balanced':
        X_train, y_train = ros.fit_resample(X_train.reshape(-1, 1), y_train)

    unique_classes_train, counts_train = np.unique(y_train, return_counts=True)
    unique_classes_test, counts_test = np.unique(y_val_test, return_counts=True)

    print("Training set class distribution:")
    for cls, count in zip(unique_classes_train, counts_train):
        print(f"Class {cls}: {count} samples")

    print("\nTesting set class distribution:")
    for cls, count in zip(unique_classes_test, counts_test):
        print(f"Class {cls}: {count} samples")

    X_val, X_test, y_val, y_test = train_test_split(X_val_test, 
                                                    y_val_test, 
                                                    test_size=0.5, 
                                                    random_state=42, 
                                                    stratify=y_val_test)

    #X_train.to_csv(f'./experiment_data/{data}/training_data/X_train.csv', index=False, header=False)
    #print("SUCCESSS")
    X_train = pd.DataFrame(X_train)
    X_train.to_csv(f'./experiment_data/{data}/{column}/{balance}/training_data/X_train.csv', header=[column], index=False)
    print("File saved: X_train")

    y_train= pd.DataFrame(y_train)
    y_train.to_csv(f'./experiment_data/{data}/{column}/{balance}/training_data/y_train.csv', header='TilldeladBeredningsgruppKortNamn', index=False)
    print("File saved: y_train")

    X_val = pd.DataFrame(X_val)
    X_val.to_csv(f'./experiment_data/{data}/{column}/{balance}/validation_data/X_val.csv', header=[column], index=False)
    print("File saved: X_val")

    y_val = pd.DataFrame(y_val)
    y_val.to_csv(f'./experiment_data/{data}/{column}/{balance}/validation_data/y_val.csv', header='TilldeladBeredningsgruppKortNamn', index=False)
    print("File saved: y_val")

    X_test = pd.DataFrame(X_test)
    X_test.to_csv(f'./experiment_data/{data}/{column}/{balance}/test_data/X_test.csv', header=[column], index=False)
    print("File saved: x_test")
   
    y_test = pd.DataFrame(y_test)
    y_test.to_csv(f'./experiment_data/{data}/{column}/{balance}/test_data/y_test.csv', header='TilldeladBeredningsgruppKortNamn', index=False)
    print("File saved: y_test")
    
    #np.savetxt(f'./experiment_data/{data}/training_data/X_train.csv', X_train_resampled, delimiter=',')
   

    #np.savetxt(f'./experiment_data/{data}/training_data/y_train.csv', y_train_resampled, delimiter=',')

    #np.savetxt(f'./experiment_data/{data}/validation_data/X_val.csv', X_val, delimiter=',')

    #np.savetxt(f'./experiment_data/{data}/validation_data/y_val.csv', y_val, delimiter=',')

    #np.savetxt(f'./experiment_data/{data}/test_data/X_test.csv', X_test, delimiter=',')

    #np.savetxt(f'./experiment_data/{data}/test_data/y_test.csv', y_test, delimiter=',')
