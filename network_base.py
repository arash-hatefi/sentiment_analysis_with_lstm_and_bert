import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



class NetworkBase(nn.Module):


    def __init__(self, device):

        self.device=device
        
        super(NetworkBase, self).__init__()
        
        self.__train_log = []
        self.__test_log = []
        
        
    def train_model(self, trainloader, optimizer, epoch, batch_size=128, shuffle=True, print_results=True):
        
        self.train()
        train_loss = 0
        true_guesses = 0
        n_batches = int(len(trainloader.dataset) / batch_size) if batch_size!=None else 1
        
        batch_log = []

        for n_batch, (data, target) in enumerate(trainloader.get_batched_data(batch_size=batch_size, shuffle=shuffle)):
            
            data = data.to(self.device)
            target = target.to(self.device)
            optimizer.zero_grad()
            output = self.__call__(data)

            loss = F.cross_entropy(output, target, reduction='sum')
            total_batch_loss = loss.item()
            train_loss += total_batch_loss
            loss.backward()
            optimizer.step()

            batch_true_guesses = sum([(torch.argmax(output[i])==target[i]).item() for i in range(len(output))])
            true_guesses += batch_true_guesses
            batch_accuracy = batch_true_guesses/len(data)
            
            batch_log.append((total_batch_loss/len(data), batch_accuracy))

            if print_results:
                print('\rTraining Batch {}/{}:\tAverage Loss: {:.3f}\tAccuracy on Batch Data: {:.3f}'.format(n_batch+1, n_batches, total_batch_loss/len(data), batch_accuracy), end='')

        self.__train_log.append((epoch, batch_log))
   
        torch.cuda.empty_cache()

        average_batch_loss = train_loss / len(trainloader.dataset) 
        accuracy = true_guesses / len(trainloader.dataset) 
        return (average_batch_loss, accuracy)
            
         
    def test_model(self, testloader, epoch, batch_size=128, print_results=True):

        self.eval()
        test_loss = 0
        true_guesses = 0
        n_batches = int(len(testloader.dataset) / batch_size) if batch_size!=None else 1
        with torch.no_grad():
          for n_batch, (data, target) in enumerate(testloader.get_batched_data(batch_size=batch_size, shuffle=False)):
              
              data = data.to(self.device)
              target = target.to(self.device)
              output = self.__call__(data)
              
              loss = F.cross_entropy(output, target, reduction='sum')
              total_batch_loss = loss.item()
              test_loss += total_batch_loss
              
              batch_true_guesses = sum([(torch.argmax(output[i])==target[i]).item() for i in range(len(output))])
              true_guesses += batch_true_guesses
              batch_accuracy = batch_true_guesses/len(data)
              
              if print_results:
                  print('\rTesting Batch {}/{}:\tTotal Batch Loss Loss: {:.3f}\tAccuracy on Batch Data: {:.3f}'.format(n_batch+1, n_batches, total_batch_loss/len(data), batch_accuracy), end='')

        torch.cuda.empty_cache() 
        
        average_batch_loss = test_loss / len(testloader.dataset) 
        accuracy = true_guesses / len(testloader.dataset) 
        results = (average_batch_loss, accuracy)

        self.__test_log.append((epoch, results))

        return results


    def get_train_log(self):
        
        return self.__train_log
    
    
    def get_test_log(self):
        
        return self.__test_log
    
    
    def reset_log(self):
        
        self.__train_log = []
        self.__test_log = []
        
        
    def get_loss_plots(self, figsize=(10,5), dpi=200):

        assert self.__train_log, "No trained model!"

        train_epochs = [epoch_number+batch_idx/len(epoch_info[1]) for (epoch_number, epoch_info) in enumerate(self.__train_log) for (batch_idx, _) in enumerate(epoch_info[1])] 
        train_loss = [average_batch_loss for epoch_info in self.__train_log for average_batch_loss,_ in epoch_info[1]]

        test_epochs = [epoch_number+1 for epoch_number, _ in enumerate(self.__test_log)]
        test_loss = [epoch_info[1][0] for epoch_info in self.__test_log]

        plt.figure(figsize=figsize, dpi=dpi)
        plt.plot(train_epochs, train_loss, color='green', label='Average Loss on train data')
        plt.scatter(test_epochs, test_loss, color='blue', label='Average Loss on test data', zorder=2.5)
        plt.xlabel('Epoch')
        plt.ylabel('Average Loss')
        plt.legend()
        plt.show()

  
    def get_accuracy_plots(self, figsize=(10,5), dpi=200):

        assert self.__train_log, "No trained model!"

        train_epochs = [epoch_number+batch_idx/len(epoch_info[1]) for (epoch_number, epoch_info) in enumerate(self.__train_log) for (batch_idx, _) in enumerate(epoch_info[1])] 
        train_accuracy = [batch_accuracy for epoch_info in self.__train_log for _, batch_accuracy in epoch_info[1]]
        
        test_epochs = [epoch_number+1 for epoch_number, _ in enumerate(self.__test_log)]
        test_accuracy = [epoch_info[1][1] for epoch_info in self.__test_log]

        plt.figure(figsize=figsize, dpi=dpi)
        plt.plot(train_epochs, train_accuracy, color='green', label='Accuracy on train data')
        plt.scatter(test_epochs, test_accuracy, color='blue', label='Accuracy on test data', zorder=2.5)
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.show()


    def get_confusion_matrix(self, dataloader, batch_size=128):

        self.eval()
        target_list = []
        output_list = []
        n_batches = int(len(dataloader.dataset) / batch_size) if batch_size!=None else 1
        with torch.no_grad():
            for n_batch, (data, target) in enumerate(dataloader.get_batched_data(batch_size=batch_size)):
                data = data.to(self.device)
                target = target.to(self.device)
                output = self.__call__(data)
                output_list.extend(np.argmax(output.tolist(),1))
                target_list.extend(target.tolist())
                print('\rBatch {}/{}'.format(n_batch+1, n_batches), end='')

            print('\r')
        target_and_output = list(zip(output_list, target_list))

        confusion_matrix = pd.DataFrame(data=[[target_and_output.count((0,0)),
                                               target_and_output.count((1,0)),
                                               target_and_output.count((2,0))],
                                              [target_and_output.count((0,1)),
                                               target_and_output.count((1,1)),
                                               target_and_output.count((2,1))],
                                              [target_and_output.count((0,2)),
                                               target_and_output.count((1,2)),
                                               target_and_output.count((2,2))]], 
                                       columns=['Predicted Positive', 'Predicted Natural', 'Predicted Negative'], 
                                       index=['Actual Positive', 'Actual Natural', 'Actual Negative'])
        
        return confusion_matrix