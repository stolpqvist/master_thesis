#TODO To make this class be able to handle and coordinate models
#TODO To handle the different types of trainers
#TODO To handle file writing and plot creation
#TODO To handle parameter hunting for various models
#



class ExperimentOrganiser:

    def __init__(self, *args, **kwargs):
        self.model = model
        if self.model.lower() in ['roberta', 'rnn', 'cnn']:
            if self.model.lower() != 'roberta':
                from .preprocessing.pre_nn import 
                trainer =
        else:
            print('Not model available for selection')
            break
        
