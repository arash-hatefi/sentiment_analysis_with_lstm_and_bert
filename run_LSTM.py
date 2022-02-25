from networks import UnidirectionalLSTM, BidirectionalLSTM, PyramidalLSTM
from dataloaders import LSTMDataloader
import pickle
import torch
from torch import optim
import pandas as pd

##############################################################################
############################ HYPER-PARAMETERS ################################
##############################################################################

# Embedding Dictionary and Dataset Path
EMBEDDING_DICT_PATH = './embedding_dict.pickle'
DATASET_PATH = './processed_dataset.pickle'

# Train, Test, and Validation Proportions
TRAIN_DATA_PROPORTION = 0.89
TEST_DATA_PROPORTION = 0.1
VALIDATION_DATA_PROPORTION = 0.01

# Model to Train: Change the name to UnidirectionalLSTM, BidirectionalLSTM, PyramidalLSTM to train other models
MODEL = UnidirectionalLSTM

# Whether or Not To Use GPU for Training
USE_GPU = True

# Input Sequence Length to the Network
SEQUENCE_LENGTH = 280

# Learning Parameters
LEARNING_RATE = 1e-3
EPOCHS = 10
BATCH_SIZE = 25
LOG_EVERY = 1




##############################################################################
############################# PREPARING DATASET ##############################
##############################################################################

def load_pickle(path):
    file = open(path, 'rb')
    return_file = pickle.load(file)
    file.close()
    return return_file

embedding_dict = load_pickle(EMBEDDING_DICT_PATH)

dataset = load_pickle(DATASET_PATH).reset_index().sample(frac=1)

dataset = list(zip(dataset['basic_tokenized_text'].values.tolist(), 
                   dataset['polarity'].values.tolist()))

train_set = dataset[0:int(TRAIN_DATA_PROPORTION*len(dataset))]
test_set = dataset[int(TRAIN_DATA_PROPORTION*len(dataset)):int((TRAIN_DATA_PROPORTION+TEST_DATA_PROPORTION)*len(dataset))]
validation_set = dataset[int((TRAIN_DATA_PROPORTION+TEST_DATA_PROPORTION)*len(dataset)):int((TRAIN_DATA_PROPORTION+TEST_DATA_PROPORTION+VALIDATION_DATA_PROPORTION)*len(dataset))]

trainloader = LSTMDataloader(dataset=train_set, embedding_dict=embedding_dict, sequence_length=SEQUENCE_LENGTH)
testloader = LSTMDataloader(dataset=test_set, embedding_dict=embedding_dict, sequence_length=SEQUENCE_LENGTH)
validationloader = LSTMDataloader(dataset=validation_set, embedding_dict=embedding_dict, sequence_length=SEQUENCE_LENGTH)




##############################################################################
############################# BUILDING THE MODEL #############################
##############################################################################

device = torch.device("cuda:0" if torch.cuda.is_available() and USE_GPU else "cpu")
    
model = MODEL(device)
model.reset_log()

optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
 



##############################################################################
############################# TRAINING THE MODEL #############################
##############################################################################

for epoch in range(1, EPOCHS+1):
    
    print_results=False
    if not epoch%LOG_EVERY:
        print_results=True
        print("\nEpoch {}:".format(epoch))    
    train_results = model.train_model(trainloader=trainloader, optimizer=optimizer, epoch=epoch, batch_size=BATCH_SIZE, shuffle=True, print_results=print_results)
    test_results = model.test_model(testloader=testloader, epoch=epoch, batch_size=BATCH_SIZE, print_results=print_results)

    if print_results:
        print("\r\tAverage Train-Set Loss: {:.3f}\tTrain-Set Accuracy: {:.3f}".format(*train_results))
        print("\r\tAverage Test-Set Loss: {:.3f}\tTest-Set Accuracy: {:.3f}".format(*test_results))  
        
        
      

##############################################################################
############################# SHOWING THE RESULTS ############################
##############################################################################
        
model.get_loss_plots()
model.get_accuracy_plots()
model.get_confusion_matrix(validationloader)
