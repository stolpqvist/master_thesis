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



