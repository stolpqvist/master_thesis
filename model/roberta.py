import torch
from transformers import AutoTokenizer, RobertForSequenceClassification



tokeniser = AutoTokenizer.from_pretrained('FacebookAI/roberta-base')
model = RobertForSequenceClassification.from_pretrained('FacebookAI/roberta-base')



num_labels(model.config.id2label)
model = XLMRobertaForSequenceClassification.from_pretrained(
        'FacebookAI/roberta-base',
        num_labels = num_labels,
        problem_type = 'multi-label-classification',
        lables= our_labels
        )



class Robsterbobster():
    def __init__(self, labels):
        self.labels = labels
        
#What we could do
#Build a custom class for the roberta
#In the class:
    # Run through the process of setting it up
    # Fix the vocabulary, unless it is shared amongst other classes
    # At which point we would create a separate class for it

