from networks import BertModel
from dataloaders import BertDataloader
import pickle
import torch
from torch import optim
import pandas as pd

##############################################################################
############################ HYPER-PARAMETERS ################################
##############################################################################

# Embedding Dictionary and Dataset Path
DATASET_PATH = './processed_dataset.pickle'

# Train, Test, and Validation Proportions
TRAIN_DATA_PROPORTION = 0.89
TEST_DATA_PROPORTION = 0.1
VALIDATION_DATA_PROPORTION = 0.01

# Model to Train
MODEL = BertModel

# Whether or Not To Use GPU for Training
USE_GPU = True

# Input Sequence Length to the Network
SEQUENCE_LENGTH = 300

# Learning Parameters
LEARNING_RATE = 3e-6
EPOCHS = 2
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

dataset = load_pickle(DATASET_PATH).reset_index().sample(frac=1)

dataset = list(zip(dataset['bert_tokenized_text_id'].values.tolist(), 
                   dataset['polarity'].values.tolist()))

train_set = dataset[0:int(TRAIN_DATA_PROPORTION*len(dataset))]
test_set = dataset[int(TRAIN_DATA_PROPORTION*len(dataset)):int((TRAIN_DATA_PROPORTION+TEST_DATA_PROPORTION)*len(dataset))]
validation_set = dataset[int((TRAIN_DATA_PROPORTION+TEST_DATA_PROPORTION)*len(dataset)):int((TRAIN_DATA_PROPORTION+TEST_DATA_PROPORTION+VALIDATION_DATA_PROPORTION)*len(dataset))]

trainloader = BertDataloader(dataset=train_set, sequence_length=SEQUENCE_LENGTH)
testloader = BertDataloader(dataset=test_set, sequence_length=SEQUENCE_LENGTH)
validationloader = BertDataloader(dataset=validation_set, sequence_length=SEQUENCE_LENGTH)




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
