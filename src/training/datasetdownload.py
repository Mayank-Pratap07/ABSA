import pandas as pd
from datasets import load_dataset
from transformers import DistilBertTokenizer

# Loading the Amazon Polarity dataset
amazon_review = load_dataset('amazon_polarity')

# Extraction of the train and test splits
train_data = amazon_review['train'].shuffle(seed=42).select(range(100000))   #Selecting 100K since dataset was too big
test_data = amazon_review['test'].shuffle(seed=42).select(range(25000))      #Selecting 25K since dataset was too big

# Loading the DistilBERT tokenizer
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

# Defining a function to tokenize the dataset
def tokenize_function(examples):
    return tokenizer(examples['content'], truncation=True, padding='max_length', max_length=128)

# Tokenizing the train and test datasets
token_train = train_data.map(tokenize_function, batched=True)
token_test = test_data.map(tokenize_function, batched=True)

# Saving the tokenized datasets to the local folder
token_train.save_to_disk('C:/Boardgames_ABSA/data/processed/train_data')
token_test.save_to_disk('C:/Boardgames_ABSA/data/processed/test_data')

print(token_train[0])
print("Tokenization complete. Datasets saved to disk.")
