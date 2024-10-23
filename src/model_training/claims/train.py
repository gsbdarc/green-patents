"""
Load a dataset and train a BERT classifier using pre-tokenized claims data.
"""
from glob import glob
import pandas as pd
import numpy as np
import torch
import ast
import os
import yaml
from torch.utils.data import Dataset, DataLoader
from transformers import BertForSequenceClassification 
from torch.optim import AdamW
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import NeptuneLogger
import torchmetrics

# Configurable paths
PROJ_DIR = '/zfs/projects/faculty/jinhwan-green-patents'

# Step 1: Load Configuration
with open(f'{PROJ_DIR}/config.yaml', 'r') as file:
    config = yaml.safe_load(file)

# Extract configuration parameters
SECTION = config['section']  # e.g., 'F'
YEAR_THRESHOLD = config['year']  # e.g., 2010

# Data Class
class ChunkDataset(Dataset):
    def __init__(self, tokenized_chunks, labels):
        self.tokenized_chunks = tokenized_chunks
        self.labels = labels

    def __len__(self):
        return len(self.tokenized_chunks)

    def __getitem__(self, idx):
        # Convert the string representation of the list back to a list of integers
        input_ids = ast.literal_eval(self.tokenized_chunks[idx])

        # Create attention masks (1 for non-padding tokens, 0 for padding tokens)
        attention_mask = [1 if token_id != 0 else 0 for token_id in input_ids]

        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }

# Model Class
class BertClassifier(pl.LightningModule):
    def __init__(self, num_labels=2):
        super().__init__()
        self.model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=num_labels)
        self.train_accuracy = torchmetrics.Accuracy(task='binary')
        self.val_accuracy = torchmetrics.Accuracy(task='binary')
        self.train_f1 = torchmetrics.F1Score(num_classes=num_labels, task='binary', average='binary')
        self.val_f1 = torchmetrics.F1Score(num_classes=num_labels, task='binary', average='binary')

    def forward(self, input_ids, attention_mask, labels):
        return self.model(input_ids, attention_mask=attention_mask, labels=labels)

    def training_step(self, batch, batch_idx):
        outputs = self(batch['input_ids'], batch['attention_mask'], batch['labels'])
        loss = outputs.loss
        preds = torch.argmax(outputs.logits, dim=1)
        labels = batch['labels']
        acc = self.train_accuracy(preds, labels)
        f1 = self.train_f1(preds, labels)

        # Log metrics
        self.log('train_accuracy', acc, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_f1', f1, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def configure_optimizers(self):
        return AdamW(self.parameters(), lr=2e-5)

    def validation_step(self, batch, batch_idx):
        outputs = self(batch['input_ids'], batch['attention_mask'], batch['labels'])
        val_loss = outputs.loss
        preds = torch.argmax(outputs.logits, dim=1)
        labels = batch['labels']
        acc = self.val_accuracy(preds, labels)
        f1 = self.val_f1(preds, labels)

        # Log metrics
        self.log('val_accuracy', acc, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_f1', f1, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_loss', val_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return {'val_loss': val_loss}

    def on_validation_epoch_end(self):
        val_loss_mean = self.trainer.callback_metrics.get('val_loss')
        if val_loss_mean is not None:
            self.log('avg_val_loss', val_loss_mean, on_epoch=True, prog_bar=True, logger=True)

# Input directory for tokenized train and val claims files
input_dir = f"{PROJ_DIR}/data/processed_data/claims/{SECTION}-{YEAR_THRESHOLD}"

train_file = os.path.join(input_dir, f'train_{YEAR_THRESHOLD}_claims_tokenized.csv')
val_file = os.path.join(input_dir, f'val_{YEAR_THRESHOLD}_claims_tokenized.csv')

# Directory to save and load checkpoints
checkpoint_dir = f"{PROJ_DIR}/src/model_training/claims/model-checkpoint"

# Ensure the checkpoint directory exists
os.makedirs(checkpoint_dir, exist_ok=True)
print(f"Using checkpoint directory: {checkpoint_dir}")

# Read train/val data
train_df = pd.read_csv(train_file)
val_df = pd.read_csv(val_file)

print(f"train_df.shape: {train_df.shape}")
print(f"val_df.shape: {val_df.shape}")

# Ensure that 'tokenized_chunk' and 'label' columns exist
if 'tokenized_chunk' not in train_df.columns or 'label' not in train_df.columns:
    raise ValueError("Train dataframe must contain 'tokenized_chunk' and 'label' columns.")
if 'tokenized_chunk' not in val_df.columns or 'label' not in val_df.columns:
    raise ValueError("Validation dataframe must contain 'tokenized_chunk' and 'label' columns.")

# Convert DataFrame columns to lists
train_tokenized_chunks = train_df['tokenized_chunk'].tolist()
train_labels = train_df['label'].astype(int).tolist()

val_tokenized_chunks = val_df['tokenized_chunk'].tolist()
val_labels = val_df['label'].astype(int).tolist()

# Create the Datasets
train_dataset = ChunkDataset(train_tokenized_chunks, train_labels)
val_dataset = ChunkDataset(val_tokenized_chunks, val_labels)

# DataLoaders
batch_size = 142
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=10)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, num_workers=10)

# Check for existing checkpoint and load if available

checkpoint_pattern = os.path.join(checkpoint_dir, '*.ckpt')
checkpoints = glob(checkpoint_pattern)
if checkpoints:
    # If there are checkpoints, sort them by modification time
    latest_checkpoint = max(checkpoints, key=os.path.getmtime)
    print(f"Found checkpoint: {latest_checkpoint}")
    # Load the model from the checkpoint
    model = BertClassifier.load_from_checkpoint(latest_checkpoint)
    ckpt_path = latest_checkpoint
else:
    print("No checkpoint found. Starting training from scratch.")
    # Initialize the model
    model = BertClassifier()
    ckpt_path = None  # Start from scratch

# Checkpoint callback
checkpoint_callback = ModelCheckpoint(
    monitor='val_loss',
    dirpath=checkpoint_dir,
    filename='bert-{epoch:02d}-{train_f1:.2f}-{val_loss:.2f}-{val_f1:.2f}',
    save_top_k=5,
    mode='min',
    save_last=True
)

# Check if GPU is available
if torch.cuda.is_available():
    print("GPU is available.")
else:
    print("GPU is not available.")

# Trainer
trainer = pl.Trainer(
    accelerator='gpu',
    devices=1 if torch.cuda.is_available() else 0,
    max_epochs=3,  # Train for 1 epoch
    callbacks=[checkpoint_callback],
#    log_every_n_steps=10,
)

# Modify trainer.fit() to include ckpt_path
trainer.fit(model, train_dataloader, val_dataloader, ckpt_path=ckpt_path)

print("Done training.")
# Load the best model from checkpoint
best_model_path = checkpoint_callback.best_model_path
print(f"Best model path: {best_model_path}")
