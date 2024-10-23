import pandas as pd
import numpy as np
import torch
import os
import time
import ast
import yaml
from torch.utils.data import Dataset, DataLoader
from transformers import BertForSequenceClassification
import pytorch_lightning as pl
import torchmetrics
import torch.nn.functional as F

# Configurable paths
PROJ_DIR = '/zfs/projects/faculty/jinhwan-green-patents'

# Load Configuration
with open(f'{PROJ_DIR}/config.yaml', 'r') as file:
    config = yaml.safe_load(file)

# Extract configuration parameters
SECTION = config['section']  # e.g., 'F'
YEAR_THRESHOLD = config['year']  # e.g., 2010

# Input directory for details text files
input_dir = f"{PROJ_DIR}/data/processed_data/tokenized_details/{SECTION}-{YEAR_THRESHOLD}"

# Path to the post-year details file
post_year_details_file = os.path.join(input_dir, f'post_{YEAR_THRESHOLD}_details_tokenized.csv')

# Output directory for predictions
output_dir = f"{PROJ_DIR}/predictions/details-model/"
os.makedirs(output_dir, exist_ok=True)

# Model path
model_path = f"{PROJ_DIR}/src/model_training/details/model-checkpoint/last.ckpt"

# Use GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# Model Class
class BertClassifier(pl.LightningModule):
    def __init__(self, num_labels=2):
        super().__init__()
        self.model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=num_labels)
        self.train_accuracy = torchmetrics.Accuracy(task='binary')
        self.val_accuracy = torchmetrics.Accuracy(task='binary')
        self.train_f1 = torchmetrics.F1Score(num_classes=num_labels, task='binary', average='binary')
        self.val_f1 = torchmetrics.F1Score(num_classes=num_labels, task='binary', average='binary')

    def forward(self, input_ids, attention_mask, labels=None):
        return self.model(input_ids, attention_mask=attention_mask, labels=labels)

    def test_step(self, batch, batch_idx):
        # Move batch to device
        batch = {k: v.to(self.device) for k, v in batch.items()}
        # Forward pass
        outputs = self(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask']
        )
        # Get predictions and probabilities
        logits = outputs.logits
        preds = torch.argmax(logits, dim=1)
        probs = torch.softmax(logits, dim=1)
        # Return predictions and probabilities
        return {'preds': preds.cpu(), 'probs': probs.cpu()}

# Data Class
class ChunkDataset(Dataset):
    def __init__(self, tokenized_chunks, doc_ids, chunk_ids, labels, years):
        self.tokenized_chunks = tokenized_chunks
        self.doc_ids = doc_ids
        self.chunk_ids = chunk_ids
        self.labels = labels
        self.years = years

    def __len__(self):
        return len(self.tokenized_chunks)

    def __getitem__(self, idx):
        # Convert the string representation of the list back to a list of integers
        input_ids = ast.literal_eval(self.tokenized_chunks[idx])
        # Create attention masks
        attention_masks = [int(token_id > 0) for token_id in input_ids]
        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_masks, dtype=torch.long),
            'doc_id': self.doc_ids[idx],
            'chunk_id': self.chunk_ids[idx],
            'label': self.labels[idx],
            'year': self.years[idx],
        }

def predict_on_dataset(details_file, output_file):
    print(f"Processing dataset: {details_file}")
    # Check if output file exists and load already processed pgpub_ids and chunk_ids
    if os.path.exists(output_file):
        print(f"Output file {output_file} exists. Loading already processed pgpub_ids and chunk_ids...")
        processed_df = pd.read_csv(output_file, usecols=['pgpub_id', 'chunk_id'])
        # Combine 'pgpub_id' and 'chunk_id' as tuples to uniquely identify each record
        processed_pairs = set(zip(processed_df['pgpub_id'], processed_df['chunk_id']))
        print(f"Loaded {len(processed_pairs)} already processed records.")
    else:
        print(f"Output file {output_file} does not exist. Starting fresh.")
        processed_pairs = set()

    # Load the specified CSV file
    df = pd.read_csv(details_file)

    # Extract required columns
    doc_ids = df['pgpub_id'].tolist()
    chunk_ids = df['chunk_id'].tolist()
    # Handle labels if they exist; if not, create dummy labels
    if 'label' in df.columns:
        labels = df['label'].astype(int).tolist()
    else:
        labels = [0] * len(df)  # Dummy labels
    years = df['year'].tolist()
    tokenized_chunks_list = df['tokenized_chunk'].tolist()

    # Filter out already processed records
    print("Filtering out already processed records...")
    indices_to_keep = []
    for idx, (pgpub_id, chunk_id) in enumerate(zip(doc_ids, chunk_ids)):
        if (pgpub_id, chunk_id) not in processed_pairs:
            indices_to_keep.append(idx)

    print(f"Total records before filtering: {len(doc_ids)}")
    print(f"Total records after filtering: {len(indices_to_keep)}")

    # If all records have been processed, exit the function
    if not indices_to_keep:
        print("All records have been processed for this dataset.")
        return

    # Update the data lists to only include unprocessed records
    doc_ids = [doc_ids[idx] for idx in indices_to_keep]
    chunk_ids = [chunk_ids[idx] for idx in indices_to_keep]
    labels = [labels[idx] for idx in indices_to_keep]
    years = [years[idx] for idx in indices_to_keep]
    tokenized_chunks_list = [tokenized_chunks_list[idx] for idx in indices_to_keep]

    # Create the dataset and dataloader
    dataset = ChunkDataset(tokenized_chunks_list, doc_ids, chunk_ids, labels, years)
    dataloader = DataLoader(dataset, batch_size=256, num_workers=10)

    print(f"Length of dataloader: {len(dataloader)}")

    # Load the trained model checkpoint
    model = BertClassifier()

    # Load the model onto the device
    model.load_state_dict(torch.load(model_path, map_location=device)['state_dict'])
    model.to(device)
    model.eval()  # Set the model to evaluation mode

    start_time = time.time()

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            print(f"Processing batch {batch_idx + 1}/{len(dataloader)}")
            # Move batch to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            # Forward pass
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            probs = torch.softmax(logits, dim=1).cpu().numpy()[:, 1]  # Assuming class 1 is the positive class

            # Get the other data from the batch
            batch_doc_ids = batch['doc_id']
            batch_chunk_ids = batch['chunk_id']
            batch_labels = batch['label']
            batch_years = batch['year']

            # Create a DataFrame for the batch
            batch_results_df = pd.DataFrame({
                'pgpub_id': batch_doc_ids,
                'chunk_id': batch_chunk_ids,
                'label': batch_labels,
                'year': batch_years,
                'predictions': preds,
                'probabilities': probs
            })

            # Save the batch results to CSV file in append mode
            if not os.path.exists(output_file) and batch_idx == 0:
                # If the file doesn't exist and this is the first batch, write headers
                batch_results_df.to_csv(output_file, index=False, mode='w', header=True)
            else:
                # Append to the file without headers
                batch_results_df.to_csv(output_file, index=False, mode='a', header=False)

    end_time = time.time()

    # Calculate and print the total time taken
    total_time = end_time - start_time
    print(f"Total evaluation time for {details_file}: {total_time:.2f} seconds")
    print(f"Done predicting on {details_file}")

# Predict on post-year details
post_output_file = os.path.join(output_dir, f'post_{YEAR_THRESHOLD}_details_predictions.csv')
predict_on_dataset(post_year_details_file, post_output_file)
