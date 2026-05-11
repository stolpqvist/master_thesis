# master_thesis
Master's thesis at Vetenskapsrådet


This thesis has been done in collaboration with Vetenskapsrådet (The Swedish Research Council (SRC)).


The code herein trains either a CNN, RNN, or a RoBERTa model to automatically classify research grant proposals.

## Methodology

###  Data
    The data is provided by the SRC and is subject to privacy.
    However, the approach for this data was to split the datasets into k-folds with stratification.

### Models
    - RNN:
        An RNN with an attention-layer

    - CNN:
        A CNN with 4 parallel layers.

    - RoBERTa:
        An xlm-roberta-base model.
### Evaluation
    The code runs a bootstrap against chance and a pairwise comparison. To account for FWER the results
    get corrected with the Bonferroni method.

### Visualisation
    The code allows for the creation of confusion matricies, boxplots and F1-score distribution.
    Visualisation is only performed during bootstrapping.

## Installation

```bash
pip install -r requirements.txt
```

## Usage
To train the model you may:
```bash
python3 main.py --train --model cnn
```
Choose 'cnn', 'rnn', or 'roberta'. Note: training requires access to the SRC dataset, which cannot be shared due to its privacy status.


To test a model:
```bash
python3 main.py --test --model cnn 
```
Note: This also has to be given the -bg flag

The code also incorporates visualisation and emissions tracking:
```bash
python3 main.py --model cnn rnn roberta --boot --vis -em
```

The flag '--boot' enables bootstrapping, '--vis' enables visualisation, and '-em' controls emissions tracking.
### Flags
| Flag | Short | Description |
|------|-------|-------------|
| `--model` | `-m` | Model(s) to use: `cnn`, `rnn`, `roberta`. Accepts multiple only for `--boot` |
| `--train` | `-tr` | Enable training |
| `--test` | | Enable testing |
| `--boot` | `-b` | Enable bootstrapping |
| `--vis` | `-v` | Enable visualisation |
| `--emissions` | `-em` | Enable emissions tracking |
| `--param_hunt` | `-p` | Enable hyperparameter optimisation |
| `--batch_size` | `-bs` | Batch size (default: 2) |
| `-k` | | Number of k-folds (default: 10) |
| `-e` | | Number of epochs (default: 10) |
| `-lr` | | Learning rate (default: 0.00001) |
| `-dr` | | Dropout rate (default: 0.1) |
| `-bg` | | Training group |
| `--columns` | `-c` | Data columns to use |
| `--label` | `-l` | Label column |
| `--file` | `-f` | File to read from if `-bg` is not provided |



## Project Structure
```
.
├── main.py                         # Entry point
├── config.py                       # Configuration dataclass
├── experiment_handler.py           # Orchestrates experiments
├── sig_test.py                     # Bootstrap and significance testing
├── data_handling/
│   └── strat_fold.py               # Stratified k-fold cross-validation
├── model/
│   ├── cnn.py                      # CNN model
│   ├── rnn.py                      # Attention-based LSTM RNN model
│   └── roberta.py                  # XLM-RoBERTa model
├── preprocessing/
│   ├── pre_nn.py                   # Preprocessing for CNN and RNN
│   ├── pre_roberta.py              # Preprocessing for RoBERTa
│   └── tokenizers/                 # Custom tokenisers
├── train/
│   ├── train.py                    # General training logic
│   └── train_nn.py                 # CNN/RNN specific training
└── utils/
    ├── path_manager.py             # Path management
    └── visualisation.py            # Confusion matrices, boxplots, F1 plots
```
Note: Data handling and splitting utilities are omitted due to aforementioned privacy status.
