import pickle
import numpy as np
from itertools import chain 



def generate_word_embedding_dict(embedding_file_path, processed_dataset_path):

    SAVE_DIRECTORY = './'
    EMBEDDING_LENGTH = 300
    
    def load_pickle(path):
        file = open(path, 'rb')
        return_file = pickle.load(file)
        file.close()
        return return_file

    tokens_set = set(chain.from_iterable(load_pickle(processed_dataset_path)['basic_tokenized_text']))

    embedding_dict = {}
    
    file = open(embedding_file_path, 'r', encoding="utf8")
    
    last_line_number = 0
    sum_of_vectors = np.zeros(EMBEDDING_LENGTH)
    for line_number, line in enumerate(file):
        print('\rreading line number {}'.format(line_number), end='')
        word_list = line.split()
        key = word_list[0]
        value = list(map(float, word_list[1:]))
        sum_of_vectors += np.array(value)
        if key in tokens_set:
            embedding_dict[key] = value
    
    file.close()
    
    number_of_lines = last_line_number+1
    mean_of_vectors = sum_of_vectors / number_of_lines
    embedding_dict[None] = list(mean_of_vectors)
    
    save_file = open(SAVE_DIRECTORY+'/embedding_dict.pickle' ,"wb")
    pickle.dump(embedding_dict, save_file)
    print(" saved!")
    save_file.close()
    
    
if __name__ == "__main__":

    EMBEDDING_FILE_PATH = './glove.42B.300d.txt'
    PROCESSED_DATASET_PATH = './processed_dataset.pickle'
    
    generate_word_embedding_dict(EMBEDDING_FILE_PATH, PROCESSED_DATASET_PATH)