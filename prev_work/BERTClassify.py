
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
import torch
from tqdm import tqdm
import os
from transformers import BertTokenizer
from torch.utils.data import TensorDataset
import pandas as pd
from transformers import BertForSequenceClassification, BertConfig
from transformers import AutoTokenizer
import numpy as np
from transformers import AutoConfig
from transformers import AutoModel
from transformers import AdamW, get_linear_schedule_with_warmup
from sklearn.metrics import f1_score
import pickle
import random
import time
import warnings
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, classification_report
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def train(data = None, column = None, dropout = None, balance = None, learning_rate = None):
    # %%
    # Denna används i BERT klassificering
    if data == None:
        data = input("NT/HS?: ").strip().upper()
    if column == None:
        print("ProjektSammanfattningEng/Keywords/ProjectTitleEng?")
        column = input("PS/K/PTE: ").upper()
        if column == "PS":
            column = "ProjektSammanfattningEng"
        if column == "K":
            column = "Keywords"
        if column == "PTE":
            column = "ProjectTitleEng"
  

    if balance == None:
        print("balanced or imbalanced dataset")
        balance = input("B/U?:" ).upper()
        if balance == 'B':
            balance = 'balanced'
        if balance == 'U':
            balance = 'unbalanced'
    if dropout == None:
        if balance == 'balanced':
            learning_rate = 3e-05
            dropout = 0.4
        if balance == 'unbalanced':
            dropout = 0.3
            learning_rate = 5e-05

    #if learning_rate == None:
    #    learning_rate = 1e-5

    train_data = pd.read_csv(f'./experiment_data/{data}/{column}/{balance}/training_data/X_train.csv')
    train_labels = pd.read_csv(f'./experiment_data/{data}/{column}/{balance}/training_data/y_train.csv')

    validation_data = pd.read_csv(f'./experiment_data/{data}/{column}/{balance}/validation_data/X_val.csv')
    validation_labels = pd.read_csv(f'./experiment_data/{data}/{column}/{balance}/validation_data/y_val.csv')
    df_val = validation_data.join(validation_labels)


    test_data = pd.read_csv(f'./experiment_data/{data}/{column}/{balance}/test_data/X_test.csv')
    test_labels = pd.read_csv(f'./experiment_data/{data}/{column}/{balance}/test_data/y_test.csv')

    df = train_data.join(train_labels)
    #df.to_csv('df', index=False)
    #print(df)
    # %%
    train_labels.head()
    


    # %%
    df.TilldeladBeredningsgruppKortNamn.value_counts()
    #print(df.TilldeladBeredningsgruppKortNamn.value_counts())
    # %%
    possible_labels = df.TilldeladBeredningsgruppKortNamn.unique()

    label_dict = {'HS-A':0, 'HS-B':1, 'HS-C':2, 'HS-D':3, 'HS-E':4, 'HS-F':5, 'HS-I':6, 'HS-J':7, 'HS-K':8, 'NT-1':9, 'NT-10':10, 'NT-11':11, 'NT-12':12, 'NT-13':13, 'NT-14':14, 'NT-15':15, 'NT-16':16, 'NT-17':17, 'NT-18':18, 'NT-19':19, 'NT-2':20, 'NT-3':21, 'NT-4':22, 'NT-5':23, 'NT-6':24, 'NT-7':25, 'NT-8':26, 'NT-9':27}
    #for index, possible_label in enumerate(possible_labels):
    #    label_dict[possible_label] = index
    #label_dict

    # %%
    df['label'] = df.TilldeladBeredningsgruppKortNamn.replace(label_dict)
    df_val['label'] = df_val.TilldeladBeredningsgruppKortNamn.replace(label_dict)

    # %%
    df.head()

    # %%


    # %%
    df['data_type'] = ['not_set']*df.shape[0]

    #Entire  dataframe consists of training data so locating
    #the training data is unnecessary.
    #df.loc[train_data, 'data_type'] = 'train'
    #df.loc[validation_data, 'data_type'] = 'val'

    # %%
    #df.groupby(['TilldeladBeredningsgruppKortNamn', 'label', 'data_type']).count()

    # %%
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', 
                                            do_lower_case=True)
    #tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased', config = 'allenai/scibert_scivocab_uncased/config.json')
    data_list = df[column].values.tolist()
    val_list = df_val[column].values.tolist()
    #df_val[column].values

    #df[column].values
    # %%
    encoded_data_train = tokenizer.batch_encode_plus(
        data_list, 
        add_special_tokens=True, 
        return_attention_mask=True, 
        pad_to_max_length=True, 
        max_length=256, 
        return_tensors='pt'
    )

    encoded_data_val = tokenizer.batch_encode_plus(
        val_list, 
        add_special_tokens=True, 
        return_attention_mask=True, 
        pad_to_max_length=True, 
        max_length=256, 
        return_tensors='pt'
    )

    input_ids_train = encoded_data_train['input_ids']
    attention_masks_train = encoded_data_train['attention_mask']
    labels_train = torch.tensor(df.label.values)

    input_ids_val = encoded_data_val['input_ids']
    attention_masks_val = encoded_data_val['attention_mask']
    labels_val = torch.tensor(df_val.label.values)

    # %%
    dataset_train = TensorDataset(input_ids_train, attention_masks_train, labels_train)
    dataset_val = TensorDataset(input_ids_val, attention_masks_val, labels_val)

    # %%
    len(dataset_train), len(dataset_val)

    # %%
    #Configuere BertModel
    configuration = BertConfig.from_pretrained('bert-base-uncased',
                                                num_labels=len(label_dict),
                                                output_attentions=False,
                                                output_hidden_states=False,)
    #configuration = AutoConfig.from_pretrained('./allenai/scibert_scibocab_uncased',
    #                                            num_labels=len(label_dict),
    #                                            output_attentions=False,
    #                                            output_hidden_states=False,)
    configuration.hidden_dropout_prob = dropout
    configuration.attention_probs_dropout_prob = dropout

    #Initiate model
    model = BertForSequenceClassification.from_pretrained("bert-base-uncased", config=configuration)
    
    #model = AutoModel.from_pretrained('allenai/scibert_scivocab_uncased')

    # %%


    batch_size = 3

    dataloader_train = DataLoader(dataset_train, 
                                sampler=RandomSampler(dataset_train), 
                                batch_size=batch_size)

    dataloader_validation = DataLoader(dataset_val, 
                                    sampler=SequentialSampler(dataset_val), 
                                    batch_size=batch_size)

    # %%


    optimizer = AdamW(model.parameters(),
                    lr=learning_rate, 
                    eps=1e-8)

    # %%
    epochs = 50

    scheduler = get_linear_schedule_with_warmup(optimizer, 
                                                num_warmup_steps=0,
                                                num_training_steps=len(dataloader_train)*epochs)

    #Setting a random seed
    seed_val = 17
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)

# %%
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    print("Model device:", next(model.parameters()).device)

    print(device)
    f1_tally = []
    start = time.time()
    for epoch in tqdm(range(1, epochs+1)):
    
        model.train()
        
        loss_train_total = 0

        progress_bar = tqdm(dataloader_train, desc='Epoch {:1d}'.format(epoch), leave=False, disable=False)

        for batch in progress_bar:

            model.zero_grad()

            batch = tuple(b.to(device) for b in batch)

            inputs = {'input_ids':      batch[0],
                    'attention_mask': batch[1],
                    'labels':         batch[2],
                    }       
            outputs = model(**inputs)

            loss = outputs[0]
            loss_train_total += loss.item()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()
            scheduler.step()
            
            progress_bar.set_postfix({'training_loss': '{:.3f}'.format(loss.item()/len(batch))})
            
        directory = 'data_volume'   
        if not os.path.exists(directory):
            os.makedirs(directory)

        torch.save(model.state_dict(), f'data_volume/finetuned_BERT_epoch_{epoch}.model')
            
        #tqdm.write(f'\nEpoch {epoch}')
        
        loss_train_avg = loss_train_total/len(dataloader_train)            
        #tqdm.write(f'Training loss: {loss_train_avg}')
        
        val_loss, predictions, true_vals = evaluate(dataloader_validation, model)
        val_f1 = f1_score_func(predictions, true_vals)
        #tqdm.write(f'Validation loss: {val_loss}')
        #tqdm.write(f'F1 Score (Macro): {val_f1}')
        f1_tally.append(round(val_f1, 2))
    end = time.time()
    Training_time = end - start
    print(f1_tally)
    print("Hidden dropout probability:", configuration.hidden_dropout_prob)
    print("Attention dropout probability:", configuration.attention_probs_dropout_prob)
    print(f"Training took {Training_time}")
        

    # %%
    model = BertForSequenceClassification.from_pretrained("bert-base-uncased",
                                                        num_labels=len(label_dict),
                                                        output_attentions=False,
                                                        output_hidden_states=False,
                                                        )

    model.to(device)

    # %%
    model.load_state_dict(torch.load('data_volume/finetuned_BERT_epoch_50.model', map_location=torch.device('cpu')))
    with open(f"./experiment_data/{data}/{column}/{balance}/possible_labels.pickle", 'wb') as label_file:
        pickle.dump(label_dict, label_file)

    torch.save(model.state_dict(), f"./experiment_data/{data}/{column}/{balance}/bert_model.pth")
    tokenizer.save_pretrained(f"./experiment_data/{data}/{column}/{balance}/tokenizer")
    return f1_tally

def validation(data = None, column = None, balance = None):
    if data == None:
        data = input("NT/HS?: ")
    if column == None:
        print("ProjektSammanfattningEng/Keywords/ProjectTitleEng?")
        column = input("PS/K/PTE: ")
        if column == "PS":
            column = "ProjektSammanfattningEng"
        if column == "K":
            column = "Keywords"
        if column == "PTE":
            column = "ProjectTitleEng"

    #Loading previous models
    with open(f"./experiment_data/{data}/{column}/{balance}/possible_labels.pickle", 'rb') as possible_labels:
        label_dict = pickle.load(possible_labels)
    tokenizer = BertTokenizer.from_pretrained(f"./experiment_data/{data}/{column}/{balance}/tokenizer")
    model = BertForSequenceClassification.from_pretrained("bert-base-uncased")
    model.load_state_dict(torch.load(f"./experiment_data/{data}/{column}/{balance}/tokenizer"))

    validation_data = pd.read_csv(f'./experiment_data/{data}/{column}/{balance}/validation_data/X_val.csv')
    validation_labels = pd.read_csv(f'./experiment_data/{data}/{column}/{balance}/validation_data/y_val.csv')
    df_val = validation_data.join(validation_labels)

    df_val['label'] = df_val.TilldeladBeredningsgruppKortNamn.replace(label_dict)    

    encoded_data_val = tokenizer.batch_encode_plus(
    df_val[column].values, 
    add_special_tokens=True, 
    return_attention_mask=True, 
    pad_to_max_length=True, 
    max_length=256, 
    return_tensors='pt'
    )

    input_ids_val = encoded_data_val['input_ids']
    attention_masks_val = encoded_data_val['attention_mask']
    labels_val = torch.tensor(df_val.label.values)

    dataset_val = TensorDataset(input_ids_val, attention_masks_val, labels_val)

    model = BertForSequenceClassification.from_pretrained("bert-base-uncased",
                                                        num_labels=len(label_dict),
                                                        output_attentions=False,
                                                        output_hidden_states=False)

    batch_size = 3


    dataloader_validation = DataLoader(dataset_val, 
                                   sampler=SequentialSampler(dataset_val), 
                                   batch_size=batch_size)
def hp_test():
    warnings.filterwarnings("ignore")

    data = input("NT/HS: ?")
    data = data.strip().upper()
    print("ProjektSammanfattningEng/Nyckelord/AnsökanTitelEng: ?")
    column = input("P/N/A: ?")
    column = column.upper()
    all_column = ["Nyckelord", "AnsökanTitelEng"]
    learning_rates = [2e-5, 3e-5, 5e-5]
    all_dropout = [0, 0.1, 0.2, 0.3, 0.4, 0.5]
    balances = ['balanced', 'unbalanced']
    #n_column = [0.3, 0.4, 0.5]
    if column == "P":
        column = "ProjektSammanfattningEng"
    if column == "N":
        column = "Nyckelord"
    if column == "A":
        column = "AnsökanTitelEng"

    with open('BERT_test_score.txt', 'w') as file_bert:
    #f1 = train(data, column)
        if column == "ALLT":
            column = 'ALL'
            for learning_rate in learning_rates:
                for balance in balances:
                    for dropout in all_dropout:
                        f1 = train(data, column, dropout, balance, learning_rate)
                        file_bert.write(f'Learning Rate: {learning_rate}, Balance : {balance}, Dropout: {dropout}, F1: {f1}\n')
    if column == "ALL":
        for column in all_column:
            for dropout in all_dropout:
                f1 = train(data, column, dropout)
            print(column)
    if column == "NYCKEL":
        for dropout in n_column:
            train(data, "Nyckelord", dropout)
        for dropout in all_dropout:
            train(data, "AnsökanTitelEng", dropout)
    #validation(data, column)


def f1_score_func(preds, labels):
    preds_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return f1_score(labels_flat, preds_flat, average='macro')

def accuracy_per_class(preds, labels):

    with open('possible_labels.pickle', 'rb') as possible_labels:
        label_dict = pickle.load(possible_labels)

    label_dict_inverse = {v: k for k, v in label_dict.items()}
    
    preds_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()

    for label in np.unique(labels_flat):
        y_preds = preds_flat[labels_flat==label]
        y_true = labels_flat[labels_flat==label]
        print(f'Class: {label_dict_inverse[label]}')
        print(f'Accuracy: {len(y_preds[y_preds==label])}/{len(y_true)}\n')


def evaluate(dataloader_val, model):

    #model = 
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    loss_val_total = 0
    predictions, true_vals = [], []
    
    for batch in dataloader_val:
        
        batch = tuple(b.to(device) for b in batch)
        
        inputs = {'input_ids':      batch[0],
                  'attention_mask': batch[1],
                  'labels':         batch[2],
                 }

        with torch.no_grad():        
            outputs = model(**inputs)
            
        loss = outputs[0]
        logits = outputs[1]
        loss_val_total += loss.item()

        logits = logits.detach().cpu().numpy()
        label_ids = inputs['labels'].cpu().numpy()
        predictions.append(logits)
        true_vals.append(label_ids)
    
    loss_val_avg = loss_val_total/len(dataloader_val) 
    
    predictions = np.concatenate(predictions, axis=0)
    true_vals = np.concatenate(true_vals, axis=0)
            
    return loss_val_avg, predictions, true_vals

# %%
def test(data = None, column = None, balance = None):

    if data == None:
        data = input("NT/HS?: ").strip().upper()

    if column == None:
        print("ProjektSammanfattningEng/Keywords/ProjectTitleEng?")
        column = input("PS/K/PTE: ").upper()
        if column == "PS":
            column = "ProjektSammanfattningEng"
        if column == "K":
            column = "Keywords"
        if column == "PTE":
            column = "ProjectTitleEng"

    if balance == None:
        print("balanced or imbalanced dataset")
        balance = input("B/U?:" ).upper()
        if balance == 'B':
            balance = 'balanced'
        if balance == 'U':
            balance = 'unbalanced'
    
    batch_size = 3
    with open(f"./experiment_data/{data}/{column}/{balance}/possible_labels.pickle", 'rb') as possible_labels:
        label_dict = pickle.load(possible_labels)

    test_data = pd.read_csv(f'./experiment_data/{data}/{column}/{balance}/test_data/X_test.csv')
    test_labels = pd.read_csv(f'./experiment_data/{data}/{column}/{balance}/test_data/y_test.csv')

    df_test = test_data.join(test_labels)
    test_list = df_test[column].values.tolist()


    df_test['label'] = df_test.TilldeladBeredningsgruppKortNamn.replace(label_dict)
    #print('these are the uniques ',df_test['TilldeladBeredningsgruppKortNamn'].unique())
    df_test.to_csv('bert_test_labels')
    tokenizer = BertTokenizer.from_pretrained(f"./experiment_data/{data}/{column}/{balance}/tokenizer")

    encoded_data_test = tokenizer.batch_encode_plus(
    test_list, 
    add_special_tokens=True, 
    return_attention_mask=True, 
    pad_to_max_length=True, 
    max_length=256, 
    return_tensors='pt'
    )   
    #print(df_test.label.values)
    input_ids_test = encoded_data_test['input_ids']
    attention_masks_test = encoded_data_test['attention_mask']
    labels_test = torch.tensor(df_test.label.values)
   
    dataset_test = TensorDataset(input_ids_test, attention_masks_test, labels_test)
    
    dataloader_test = DataLoader(dataset_test, 
                                    sampler=SequentialSampler(dataset_test), 
                                    batch_size=batch_size)

    model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=len(label_dict))

    # Load the saved model state dictionary
    model.load_state_dict(torch.load(f"./experiment_data/{data}/{column}/{balance}/bert_model.pth", map_location=torch.device('cpu')))

    #test_dataset = MyDataset(test_data)  # Assuming you have a custom dataset class
    test_dataloader = DataLoader(df_test, batch_size=batch_size)

    # Evaluate the model on the test set
    test_loss, test_predictions, test_true_vals = evaluate(dataloader_test, model)
    test_f1 = f1_score_func(test_predictions, test_true_vals)

    print(f'Test Loss: {test_loss}')
    print(f'Test F1 Score (Macro): {test_f1}')

    with open('test_results.txt', 'w') as f:
        f.write(f'Test Loss: {test_loss}\n')
        f.write(f'Test F1 Score (Macro): {test_f1}\n')

    #test_predictions = np.argmax(test_predictions, axis=1).flatten()
    labels_flat = test_true_vals.flatten()
    desired_order = ['HS-A', 'HS-B', 'HS-C', 'HS-D', 'HS-E', 'HS-F', 'HS-I', 'HS-J', 'HS-K', 'NT-1', 'NT-10', 'NT-11', 'NT-12', 'NT-13', 'NT-14', 'NT-15', 'NT-16', 'NT-17', 'NT-18', 'NT-19', 'NT-2', 'NT-3', 'NT-4', 'NT-5', 'NT-6', 'NT-7', 'NT-8', 'NT-9']
    labels_1= list(label_dict.keys())

    test_predictions = np.argmax(test_predictions, axis=1).flatten()
    labels_flat = test_true_vals.flatten()
    report = classification_report(labels_flat, test_predictions, target_names=labels_1)
    conf_matrix =  confusion_matrix(labels_flat, test_predictions, labels=np.unique(labels_flat))
    conf_matrix_df = pd.DataFrame(conf_matrix, index=labels_1, columns=labels_1)

    f1_mirco = f1_score(labels_flat, test_predictions, average= 'micro')
    p_micro = precision_score(labels_flat, test_predictions, average= 'micro')
    r_micro = recall_score(labels_flat, test_predictions, average = 'micro')  
    with open(f'./Bilder/TEST/micro_score_BERT_{balance}.txt', 'w') as micro:
        micro.write(f"F1-score micro: {f1_mirco}\n")
        micro.write(f"Precision micro: {p_micro}\n")
        micro.write(f"Recall micro: {r_micro}")

    print(report)

    pd.set_option('display.max_columns', None)  # or 1000
    pd.set_option('display.max_rows', None)  # or 1000
    pd.set_option('display.max_colwidth', None)  # or 19
    pd.set_option('display.width', 1000)

    plt.figure(figsize=(8, 6))  # Adjust the figure size as needed
    sns.heatmap(conf_matrix_df, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title('Confusion Matrix')
    plt.savefig(f'./Bilder/TEST/confusion_matrix_{balance}.png', dpi=300)

    # Save classification report to a text file
    with open(f'./Bilder/TEST/classification_report_{balance}.txt', 'w') as f:
        f.write(report)

