import torch
from torch.utils.data import Dataset
from datasets import load_dataset
from datasets import load_from_disk

imdb_dataset = load_dataset('imdb')

train_data = imdb_dataset['train']
test_data = imdb_dataset['test']

# Loading the tokenized datasets
token_train = load_from_disk('C:/Boardgames_ABSA/data/processed/train_data2')
token_test = load_from_disk('C:/Boardgames_ABSA/data/processed/test_data2')

# Preparing labels for training and testing
train_labels = train_data['label']
test_labels = test_data['label']

class SentimentDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(self.encodings[key][idx]) for key in self.encodings.keys()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

# Creating datasets
train_dataset = SentimentDataset(token_train, train_labels)
test_dataset = SentimentDataset(token_test, test_labels)

from transformers import AlbertForSequenceClassification, Trainer, TrainingArguments

# Loading ALBERT model for binary classification
model = AlbertForSequenceClassification.from_pretrained('albert-base-v2', num_labels=2)

# Setting training arguments
training_args = TrainingArguments(
    output_dir='C:/Boardgames_ABSA/results/IMDB',
    evaluation_strategy='epoch',
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    num_train_epochs=3,
    weight_decay=0.01
)

# Trainer instance
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset
)

# Train the model
trainer.train()