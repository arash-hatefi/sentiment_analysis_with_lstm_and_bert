# Sentiment Analysis of Tweets
Project for the Deep Learning with Applications Graduate Course, University of Tehran. (Finish date: 15-05-2020)

## Trained Models
The following models were used for sentiment analysis of a set of tweets:
1- Unidirectional LSTM
2- Bidirectional LSTM
3- Pyramidal LSTM ((See Listen, Attend and Spell)[https://arxiv.org/pdf/1508.01211.pdf])
4- Google Bert ((Pytorch pre-trained Bert)[https://pypi.org/project/pytorch-pretrained-bert/] was used)


## Dataset
The [Sentiment140](http://help.sentiment140.com/) dataset was used for training the models, which contains 1,600,000 tweets extracted using the twitter api. ([Direct download](https://docs.google.com/file/d/0B04GJPshIjmPRnZManQwWEdTZjg/edit?resourcekey=0-betyQkEmWZgp8z0DFxWsHw))

## Word Embedding
[GloVe word embedding](https://nlp.stanford.edu/projects/glove/) (42B version) was used for embedding the tokens for training LSTM models. ([Direct download](http://downloads.cs.stanford.edu/nlp/data/glove.42B.300d.zip))


## Run the code
Step 1: Download the dataset and unzip it. Here, only the train file (training.1600000.processed.noemoticon.csv) with 1,600,000 tweets were used. Put the files and the codes in the same directory.

Step 2: Download the GloVe word embedding and put it in the working directory.

Step 3: Run data_preprocess.py to preprocess the dataset. It will generate a Pandas DataFrame containing the tokenized tweets required for training the networks.

step 4: Run word_embedding_process.py for generating the word embedding dictionary for words in the dataset.

Step 5: Use run-BERT.py for training Bert, or run-LSTM.py for training LSTM networks.
