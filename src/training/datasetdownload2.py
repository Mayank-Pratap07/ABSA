import pandas as pd
from transformers import AlbertTokenizer
from datasets import load_dataset

# Loading the IMDB dataset
imdb_DS = load_dataset('imdb')

# Extracting the train and test splits
train_data = imdb_DS['train']
test_data = imdb_DS['test']

# Initializing ALBERT tokenizer
tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2')

# Tokenizing the dataset
def tokenize_function(examples):
    return tokenizer(examples['text'], padding="max_length", truncation=True, max_length=128)

# Tokenizing the train and test datasets
token_train = train_data.map(tokenize_function, batched=True)
token_test = test_data.map(tokenize_function, batched=True)

# Saving the tokenized datasets to the local folder
token_train.save_to_disk('C:/Boardgames_ABSA/data/processed/train_data2')
token_test.save_to_disk('C:/Boardgames_ABSA/data/processed/test_data2')