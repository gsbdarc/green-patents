import pandas as pd
from transformers import BertTokenizer
import time
import numpy as np
import yaml
import os

# Configurable paths
PROJ_DIR = '/zfs/projects/faculty/jinhwan-green-patents'

# Step 1: Load Configuration
with open(f'{PROJ_DIR}/config.yaml', 'r') as file:
    config = yaml.safe_load(file)

# Extract configuration parameters
SECTION = config['section']  # e.g., 'F'
YEAR_THRESHOLD = config['year']  # e.g., 2010

# Input directory for claims text files
input_dir = f"{PROJ_DIR}/data/processed_data/claims/{SECTION}-{YEAR_THRESHOLD}"

# Paths to the pre-year and post-year claims files
pre_year_claims_file = os.path.join(input_dir, f'pre_{YEAR_THRESHOLD}_claims_text.csv')
post_year_claims_file = os.path.join(input_dir, f'post_{YEAR_THRESHOLD}_claims_text.csv')

# Output directory for tokenized claims
output_dir = f"{PROJ_DIR}/data/processed_data/tokenized_claims/{SECTION}-{YEAR_THRESHOLD}"
os.makedirs(output_dir, exist_ok=True)

# Max tokens is 512 in BERT models
# Shift by half overlapping chunks
OVERLAP = 256

# Initialize the tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Tokenizer's max sequence length
MAX_LENGTH = 512  # BERT's max sequence length

def tokenize_chunk_and_pad(text, max_length=512, overlap=256):
    """
    Tokenize, chunk with overlap, and return the list of tokenized chunks.
    """
    # Tokenize the text
    tokenized_text = tokenizer.encode(text, add_special_tokens=True)
    # Initialize an empty list to store chunks
    chunks = []

    # Calculate the start index for each new chunk
    chunk_starts = range(0, len(tokenized_text), max_length - overlap)

    for i in chunk_starts:
        # Extract the chunk with potential overlap
        chunk = tokenized_text[i : i + max_length]

        # Pad the chunk if it's shorter than max_length
        chunk += [tokenizer.pad_token_id] * (max_length - len(chunk))
        
        chunks.append(chunk)

    return chunks

def chunk_and_tokenize_df(df, overlap):
    """
    Process entire df with each row of text into a list of chunked tokens.
    """
    print(f"df.shape: {df.shape}")
    # List to store dictionaries for each chunk
    chunks_list = []

    # Iterate through each row in the original DataFrame
    for index, row in df.iterrows():
        # Extract the text and metadata
        original_text = row['claims_text']  # Assuming 'claims_text' is a string
        pgpub_id = row['pgpub_id']
        green_true = row['label']
        year = row['year']

        # Tokenize, chunk, and pad the text with overlap
        chunk_tokens_list = tokenize_chunk_and_pad(original_text, max_length=MAX_LENGTH, overlap=overlap)

        for chunk_tokens in chunk_tokens_list:
            # Create a dictionary for each chunk and append to the list
            chunks_list.append({
                'pgpub_id': pgpub_id,
                'label': green_true,
                'year': year,
                'tokenized_chunk': chunk_tokens
            })
            
    return pd.DataFrame(chunks_list)

def process_and_tokenize_claims_file(input_file, output_file):
    # Read in the DataFrame
    print(f"Reading data from {input_file}")
    df = pd.read_csv(input_file, dtype={'pgpub_id': str})
    print(f"Data shape: {df.shape}")

    # Tokenize and chunk the text
    start_time = time.time()

    tokenized_chunks_df = chunk_and_tokenize_df(df, OVERLAP)

    end_time = time.time()
    total_time = end_time - start_time
    print(f"Tokenization time: {total_time:.2f} seconds")

    print("Label distribution:")
    print(tokenized_chunks_df['label'].value_counts())

    # Add chunk id
    tokenized_chunks_df['chunk_id'] = range(len(tokenized_chunks_df))

    # Save the DataFrame
    print(f"Started writing to disk: {output_file}")
    start_time = time.time()

    tokenized_chunks_df.to_csv(output_file, index=False)

    end_time = time.time()
    total_time = end_time - start_time
    print(f"Writing time: {total_time:.2f} seconds")
    print(f"Tokenized data saved to {output_file}")

# Process pre-year claims file
pre_output_file = os.path.join(output_dir, f'pre_{YEAR_THRESHOLD}_claims_tokenized.csv')
process_and_tokenize_claims_file(pre_year_claims_file, pre_output_file)

# Process post-year claims file
post_output_file = os.path.join(output_dir, f'post_{YEAR_THRESHOLD}_claims_tokenized.csv')
process_and_tokenize_claims_file(post_year_claims_file, post_output_file)

print("Done.")