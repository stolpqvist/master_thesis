from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, classification_report
from joblib import dump, load
import os
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize
import nltk
from nltk import SnowballStemmer
from nltk.corpus import stopwords
import warnings
from tqdm import tqdm
from time import time
import matplotlib.pyplot as plt
import seaborn as sns
# Detta är den senaste jag har arbetat med och är vad som kommer att användas
# för SVM klassifiering.
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
#stop_words = set(stopwords.words('english')) 
import spacy #load spacy

nlp = spacy.load('en_core_web_sm')
stops = stopwords.words("english")

#SpaCy Lemmatizer
def spacy_lemmatizer(text):
    #print(text)
    #print([word.lemma_ for word in nlp(text)])
    return [word.lemma_ for word in nlp(text)]


# we need to generate the lemmas of the stop words
stop_words_str = " ".join(stops) # nlp function needs a string
stop_words_lemma = set(word.lemma_ for word in nlp(stop_words_str))

#Nltk Lemmatizer
class LemmaTokenizer:
    ignore_tokens = [',', '.', ';', ':', '"', '``', "''", '`']
    def __init__(self):
        self.wnl = WordNetLemmatizer()
    def __call__(self, doc):
        lemma_tok = [self.wnl.lemmatize(t) for t in word_tokenize(doc) if t not in self.ignore_tokens]
        #print('Words before lemma: ', doc)

        #print("Words after lemma: ", lemma_tok)
        return [self.wnl.lemmatize(t) for t in word_tokenize(doc) if t not in self.ignore_tokens]

def stemmer(words):
    stemmer = SnowballStemmer(language='english')
    tokens = nltk.word_tokenize(words)
    stemmed_tokens = [stemmer.stem(token.lower()) for token in tokens]
    #print('Words before stemming: ', words)
    #stem_tok = ' '.join([stemmer.stem(token.lower()) for token in tokens])
    #print("Words after stem: ", stem_tok)
    return stemmed_tokens
    
def train(reg = None, data = None, column = None, loss = None, iter = None, tokenizer = None, balance = None):

    #Choose either Sciences
    #HS = Humanities and Social studies
    #NT = Natural and Engineering Studies
    #data = input("HS/NT/ALL?: ").upper()
    #data = 'NT'
    #Choose what to data column to process
    #print("ProjektSammanfattningEng/Nyckelord/AnsökanTitelEng?")
    #column = input("PS/N/PTE/ALL: ")

    #if column == "PS":
    #column = "ProjektSammanfattningEng"

    #if column == "N":
    #    column = "Nyckelord"

    #if column == "PTE":
    #    column = "ProjectTitleEng"
    #reg = float(input("C = ?: "))
    #Initialize tokenizer
    #print("Lemmatization/Stemmer?:")
    #tokenizer = input("L/S?:")
    #if tokenizer == "L":
    #    tokenizer = LemmaTokenizer()
    #if tokenizer == "S":
    

    
    if balance == None:
        print("balanced or imbalanced dataset")
        balance = input("B/U?:" ).upper()
        if balance == 'B':
            balance = 'balanced'
        if balance == 'U':
            balance = 'unbalanced'

    if data == None:
        data = 'ALL'
    if column == None:
        column = 'ALL'
    
    if reg == None:
        if balance == 'balanced':
            reg = 0.5
        if balance == 'unbalanced':
            reg = 1

    if iter == None:
        iter = 2000


    if loss == None:
        if balance == 'balanced':
            loss = 'squared_hinge'

        if balance == 'unbalanced':
            loss = 'squared_hinge'

    #Choose preprocessing method
    if tokenizer == "Spacy":
        tokenizer = spacy_lemmatizer
    if tokenizer == "Stem":
        tokenizer = stemmer
    if tokenizer == "NLTK":
        tokenizer = LemmaTokenizer()
    if tokenizer == None:

        if balance == 'balanced':
            tokenizer = LemmaTokenizer()  
     
        if balance == 'unbalanced':
            tokenizer = LemmaTokenizer()
    
    
    #Initialize training data and reshape to 2D-format
    training_data = pd.read_csv(f'./experiment_data/{data}/{column}/{balance}/training_data/X_train.csv')
    training_data = training_data.values.reshape(-1, 1)
    training_data = training_data.ravel()

    #Read labels
    training_labels = pd.read_csv(f'./experiment_data/{data}/{column}/{balance}/training_data/y_train.csv')
    training_labels = training_labels.values.reshape(-1, 1)
    training_labels = training_labels.ravel()


    # Initialize the tokenizer and the TfidfVectorizer
    tfidf = TfidfVectorizer(sublinear_tf=True, norm='l2', encoding='utf-8', ngram_range=(1, 2), min_df=1, stop_words=stops, lowercase=True, analyzer='word', tokenizer=tokenizer)



    #Vectorize the data
    training_data = tfidf.fit_transform(training_data)

    #Train the model
    model = LinearSVC(penalty='l2', C=reg, max_iter=iter, loss=loss)
    model.fit(training_data, training_labels)

    #Save models for later use
    dump(tfidf, f'./experiment_data/{data}/{column}/{balance}/models/{data}_{column}_vectorizer.pkl', compress = True)
    dump(model, f"./experiment_data/{data}/{column}/{balance}/models/{data}_{column}_linear_svc_model.joblib")
    return(f'LinearSVC settings: max_iters = {model.max_iter}, loss = {model.loss}, Data = {data}, column = {column}, tokenizer = {tokenizer}, C = {model.C}')

def validation(data, column, balance):
    #data = input("HS/NT?: ")
    #print("ProjektSammanfattningEng/Nyckelord/AnsökanTitelEng?")
    #column = input("PS/N/PTE: ")
    #if column == "PS":
    #column = "ProjektSammanfattningEng"
    #data = 'NT'
    #if column == "N":
    #    column = "Nyckelord"

    #if column == "PTE":
    #    column = "ProjectTitleEng"

    #if data == "":
    #    data = input("HS/NT?: ")
    
    #Load classifier and vectorizer
    model = load(f"./experiment_data/{data}/{column}/{balance}/models/{data}_{column}_linear_svc_model.joblib")
    tfidf = load(f'./experiment_data/{data}/{column}/{balance}/models/{data}_{column}_vectorizer.pkl')

    #Read new data and reshape to fit required input
    validation_data = pd.read_csv(f'./experiment_data/{data}/{column}/{balance}/validation_data/X_val.csv')
    validation_data = validation_data.values.reshape(-1, 1)
    validation_data = validation_data.ravel()

    #Read labels
    validation_labels = pd.read_csv(f'./experiment_data/{data}/{column}/{balance}/validation_data/y_val.csv')  

    #Vectorize data
    validation_data = tfidf.transform(validation_data)
    #print("Validation set shape:", validation_data.shape)

    #Predict
    validation_prediction = model.predict(validation_data)


    #report = classification_report(validation_labels,
    #                           validation_prediction,)
    
 
    #print(f"Accuracy: {accuracy}\n F1-Score: {F_Measure}\n Precision: {Precision}\n Recall:  {Recall} \n {Confusion_Matrix}")
    #print(report)

    #labels = np.unique(validation_labels)
    #a =  confusion_matrix(validation_labels, validation_prediction, labels=labels)

    #pd.set_option('display.max_columns', None)  # or 1000
    #pd.set_option('display.max_rows', None)  # or 1000
    #pd.set_option('display.max_colwidth', None)  # or 19
    #pd.set_option('display.width', 1000)
    #print(pd.DataFrame(a, index=labels, columns=labels))
    #print(report)
    return(f1_score(validation_labels, validation_prediction, average='macro'))

def test(data = None, column = None, balance = None):
    if data == None:
        data = 'ALL'
    if column == None:
        column = 'ALL'

    if balance == None:
        print("balanced or imbalanced dataset")
        balance = input("B/U?:" ).upper()
        if balance == 'B':
            balance = 'balanced'
        if balance == 'U':
            balance = 'unbalanced'


    model = load(f"./experiment_data/{data}/{column}/{balance}/models/{data}_{column}_linear_svc_model.joblib")
    tfidf = load(f'./experiment_data/{data}/{column}/{balance}/models/{data}_{column}_vectorizer.pkl')

    test_data = pd.read_csv(f'./experiment_data/{data}/{column}/{balance}/test_data/X_test.csv')
    test_data = test_data.values.reshape(-1, 1)
    test_data = test_data.ravel()

    test_labels = pd.read_csv(f'./experiment_data/{data}/{column}/{balance}/test_data/y_test.csv')

    test_data = tfidf.transform(test_data)

    test_prediction = model.predict(test_data)

    model = load(f"./experiment_data/{data}/{column}/{balance}/models/{data}_{column}_linear_svc_model.joblib")
    tfidf = load(f'./experiment_data/{data}/{column}/{balance}/models/{data}_{column}_vectorizer.pkl')
    
    report = classification_report(test_labels,
                                   test_prediction
                                   )
  

    f1_mirco = f1_score(test_labels, test_prediction, average= 'micro')
    p_micro = precision_score(test_labels, test_prediction, average= 'micro')
    r_micro = recall_score(test_labels, test_prediction, average = 'micro')  
    with open(f'./Bilder/TEST/micro_score_svm_{balance}.txt', 'w') as micro:
        micro.write(f"F1-score micro: {f1_mirco}\n")
        micro.write(f"Precision micro: {p_micro}\n")
        micro.write(f"Recall micro: {r_micro}")

    conf_matrix =  confusion_matrix(test_labels, test_prediction, labels= np.unique(test_labels))

    pd.set_option('display.max_columns', None)  # or 1000
    pd.set_option('display.max_rows', None)  # or 1000
    pd.set_option('display.max_colwidth', None)  # or 19
    pd.set_option('display.width', 1000)
    print(pd.DataFrame(conf_matrix, index=np.unique(test_labels), columns=np.unique(test_labels)))
    conf_matrix_df = pd.DataFrame(conf_matrix, index=np.unique(test_labels), columns=np.unique(test_labels))
    print(report)
    plt.figure(figsize=(8, 6))  # Adjust the figure size as needed
    sns.heatmap(conf_matrix_df, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title('Confusion Matrix')
    plt.savefig(f'./Bilder/TEST/confusion_matrix_svm_{balance}.png', dpi=300)  # Save with higher resolution (300 dpi in this case)

    # Save classification report to a text file
    with open(f'./Bilder/TEST/classification_report_svm_{balance}.txt', 'w') as f:
        f.write(report)



def hp_test():
    warnings.filterwarnings("ignore")
    c_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 10, 20, 30, 40, 50]
    
    iters = [1000, 2000]
    loss = ['squared_hinge', 'hinge']
    data = input("NT/HS/ALL:? ").upper()
    print("Separate/All:? ")
    column = input("S/A:? ").upper()
    if column == "S":
        column = ['AnsökanTitelEng', "ProjektSammanfattningEng", "Nyckelord"]
    if column == "A":
        column = "ALL"
    tokenizers = ["NLTK", "Stem", "Spacy"]
    balance = input("B/U/A:? ").upper()
    if balance == 'B':
        balance = 'balanced'
    if balance == 'U':
        balance = 'unbalanced'
    if balance == 'A':
        balances = ['balanced', 'unbalanced']

    with open('total_tally_hp_test.txt', 'w') as total_tally:
        for balance in balances:
            for tokenizer in tokenizers:
                for loss_value in loss:
                    for iter in iters:
                        score = []
                        start1 = time()
                        for value in c_values:
                            start = time()
                            #reg = None, data = None, column = None, loss = None, iter = None, tokenizer = None, balance = None)
                            model_settings = train(value, data, column, loss_value, iter, tokenizer, balance)
                            f1 = validation(data, column, balance)
                            f1_round = round(f1, 2)
                            score.append(f1_round)
                            end = time()
                            training_time = end - start
                            print('Runtime: ', training_time)
                        end1 = time()
                        parameter_time = end1 - start1
                        print('Runtime parameter test: ', parameter_time)

                        model_settings = str(model_settings)
                        score = str(score)
                        parameter_time = str(parameter_time)

                        total_tally.write(f'Model settings: {model_settings}, Score:{score}, Training time : {parameter_time} \n')
