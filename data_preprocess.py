import pandas as pd
from nltk import RegexpTokenizer
import pickle
import re
from pytorch_pretrained_bert import BertTokenizer



def contains_url(text):
    if text.find('http') != -1:
        return 1
    return 0


def replace_mentions(text, word):
    return re.sub(r"(?:\@)\S+", word, text)


def process_dataset(dataset_path):

    HEADER = ['polarity', 'id', 'date', 'query', 'user', 'text']

    OUTPUT_PATH = './'

    raw_dataset = pd.read_csv(dataset_path, names=HEADER, encoding='latin1')

    dataset = raw_dataset[['polarity','text']].copy()

    BASIC_TOKENIZER = RegexpTokenizer(r"\w+")
    BERT_TOKENIZER = BertTokenizer.from_pretrained('bert-base-uncased')

    basic_tokenized_text_list = []
    bert_tokenized_text_list = []
    bert_tokenized_text_id_list = []

    for idx, _ in raw_dataset.iterrows():

        print('\rProcessing texts ... {:.2f}%'.format(100*idx/len(raw_dataset)), end='')

        if contains_url(dataset.at[idx, 'text']):
            dataset = dataset.drop([idx], axis=0)
            continue

        processed_text = dataset.at[idx, 'text'].lower()
        processed_text = replace_mentions(processed_text, 'user')

        basic_tokenized_text = BASIC_TOKENIZER.tokenize(processed_text)
        bert_tokenized_text = BERT_TOKENIZER.tokenize(' '.join(word for word in basic_tokenized_text))
        bert_tokenized_text_id = BERT_TOKENIZER.convert_tokens_to_ids(bert_tokenized_text)
        
        basic_tokenized_text_list.append(basic_tokenized_text)
        bert_tokenized_text_list.append(bert_tokenized_text)
        bert_tokenized_text_id_list.append(bert_tokenized_text_id)
     
    print('\rProcessing texts ... Done!')
    print('\rProcessing other information ... ', end='')

    dataset['basic_tokenized_text'] = basic_tokenized_text_list
    dataset['bert_tokenized_text'] = bert_tokenized_text_list
    dataset['bert_tokenized_text_id'] = bert_tokenized_text_id_list
    dataset['polarity'] = dataset['polarity'].replace({0:0, 2:1, 4:2})

    dataset = dataset.reset_index()

    print('Done! \nSaving... ', end='')

    file = open(OUTPUT_PATH + "/processed_dataset.pickle" ,"wb")
    pickle.dump(dataset, file)
    file.close()

    print('Done!')
    
    
if __name__ == "__main__":

    DATASET_PATH = './training.1600000.processed.noemoticon.csv'
    process_dataset(DATASET_PATH)
