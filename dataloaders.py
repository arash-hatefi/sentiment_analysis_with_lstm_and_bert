import torch
import pandas as pd



class LSTMDataloader:

    def __init__(self, dataset, embedding_dict, sequence_length, padding='zero'):
        
        self.__PADDING_FUNCS_DICT = {'zero':self.__zero_padding}

        assert type(embedding_dict)==dict, 'Expected word embedding to be a dictionary'
        
        try: self.__padding_func = self.__PADDING_FUNCS_DICT[padding]
        except KeyError: raise KeyError('Padding is not valied!')

        self.dataset = pd.DataFrame(data=dataset, columns=['data', 'label'])
        self.embedding_dict = embedding_dict
        self.__sequence_length = sequence_length
        

    def __get_embedded_word(self, word):

        try:
            return self.embedding_dict[word]

        except KeyError:
            return self.embedding_dict[None]

    
    def __shuffle_dataset(self):

        self.dataset = self.dataset.sample(frac=1)


    def __get_batchs_indices(self, batch_size, shuffle):
        
        if shuffle: self.__shuffle_dataset()
        self.dataset = self.dataset.reset_index(drop=True)
        index_list = self.dataset.index.values.tolist()
        if batch_size==None: return [index_list]
        return [index_list[i*batch_size:((i+1)*batch_size if (i+1)*batch_size<len(index_list) else None)] for i in range (len(index_list)//batch_size+1)]


    def __get_list_embedding(self, words_list):

        return [self.__get_embedded_word(word) for word in words_list]


    def __zero_padding(self, vector):

        return vector + [[0]*300]*(self.__sequence_length - len(vector))


    def __equalize_lenghts(self, vector):

        if len(vector)>self.__sequence_length:
            return vector[0:self.__sequence_length]
        else:
            return self.__padding_func(vector)


    def get_batched_data(self, batch_size, shuffle=True):

        for batch_indices in self.__get_batchs_indices(batch_size, shuffle):
            batch = self.dataset.loc[batch_indices, :]
            batch_sentences = batch['data'].values.tolist()
            batch_labels = batch['label'].values.tolist()
            batch_embedded_sentences = [self.__get_list_embedding(sentence) for sentence in batch_sentences]
            batch_embedded_sentences = [self.__equalize_lenghts(embedded_sentence) for embedded_sentence in batch_embedded_sentences]
            yield (torch.FloatTensor(batch_embedded_sentences), torch.LongTensor(batch_labels))
            
            
            
class BertDataloader:

    __PADDING_ID = 0
    __END_OF_SENTENCE_ID = 102
    __START_OF_SENTENCE_ID = 101

    def __init__(self, dataset, sequence_length):
        
        self.dataset = pd.DataFrame(data=dataset, columns=['data', 'label'])
        self.__sequence_length = sequence_length


    def __shuffle_dataset(self):

        self.dataset = self.dataset.sample(frac=1)


    def __get_batchs_indices(self, batch_size, shuffle):
        
        if shuffle: self.__shuffle_dataset()
        self.dataset = self.dataset.reset_index(drop=True)
        index_list = self.dataset.index.values.tolist()
        if batch_size==None: return [index_list]
        return [index_list[i*batch_size:((i+1)*batch_size if (i+1)*batch_size<len(index_list) else None)] for i in range (len(index_list)//batch_size+1)]


    def __add_special_chars(self, vector):

        return [self.__STAR_OF_SENTENCE_ID]+vector+[self.__END_OF_SENTENCE_ID]


    def __equalize_length(self, vector, length):
        
        if len(vector)>length-2:
            return [self.__START_OF_SENTENCE_ID] + vector[:length-2] + [self.__END_OF_SENTENCE_ID]
        return self.__add_padding([self.__START_OF_SENTENCE_ID] + vector + [self.__END_OF_SENTENCE_ID], length)
      

    def __add_padding(self, vector, length):

        return vector + [self.__PADDING_ID] * (length - len(vector))


    def get_batched_data(self, batch_size, shuffle=True):

        for batch_indices in self.__get_batchs_indices(batch_size, shuffle):
            batch = self.dataset.loc[batch_indices, :]
            batch_sentences = batch['data'].values.tolist()
            batch_sentences = [self.__equalize_length(sentence, self.__sequence_length-2) for sentence in batch_sentences]
            polarities = batch['label'].values.tolist()
            yield (torch.LongTensor(batch_sentences), torch.LongTensor(polarities))