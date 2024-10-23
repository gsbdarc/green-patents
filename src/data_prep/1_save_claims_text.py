import pandas as pd
import glob
import json
import os
import yaml

# Configurable paths
PROJ_DIR = '/zfs/projects/faculty/jinhwan-green-patents'
CLAIMS_DIR = f'{PROJ_DIR}/data/raw_data/pg_claims'

# Step 1: Load Configuration
# Assume the YAML file is named 'config.yaml'
with open(f'{PROJ_DIR}/config.yaml', 'r') as file:
    config = yaml.safe_load(file)

# Extract configuration parameters
SECTION = config['section']                 # e.g., 'F'
YEAR_THRESHOLD = config['year']             # e.g., 2010

# Load section-year data to process
input_dir = f"{PROJ_DIR}/data/data_prep/{SECTION}-{YEAR_THRESHOLD}"
pre_year_file = os.path.join(input_dir, f'pre_{YEAR_THRESHOLD}_pgpub.csv')
post_year_file = os.path.join(input_dir, f'post_{YEAR_THRESHOLD}_pgpub.csv')

# Load the pre-year and post-year data
pre_year_df = pd.read_csv(pre_year_file, dtype={'pgpub_id': str})
post_year_df = pd.read_csv(post_year_file, dtype={'pgpub_id': str})

def match_patent_id(df_claim, pgpub_ids):
    """
    Return a dictionary with claims by matching pgpub_id in pgpub_ids
    """
    d_return = {}

    # Filter df_claim based on pgpub_id in pgpub_ids
    df_tmp = df_claim[df_claim['pgpub_id'].isin(pgpub_ids)]
    print(f"df temp shape: {df_tmp.shape}")

    # Group df_claim by pgpub_id
    grouped_df_claim = df_tmp.groupby('pgpub_id')

    # Iterate over each group
    for pgpub_id, group in grouped_df_claim:
        claims = group.set_index('claim_sequence')['claim_text'].to_dict()
        d_return[pgpub_id] = claims

    return d_return

# Claims directory and files
claims_files = glob.glob(f'{CLAIMS_DIR}/*.tsv')
print(f'Number of claims files: {len(claims_files)}')

def process_claims_for_dataset(dataset_df, dataset_name):
    """
    Processes claims for the given dataset DataFrame and returns the DataFrame with claims text added.
    """
    pgpub_ids = set(dataset_df['pgpub_id'])  # 'pgpub_id's are already strings
    print(f"Number of pgpub_ids in {dataset_name}: {len(pgpub_ids)}")

    d_claims = {}

    for f in claims_files:
        print(f"Processing file: {f}")
        df_claim = pd.read_csv(f, delimiter='\t')

        # Ensure 'pgpub_id' is a string
        df_claim['pgpub_id'] = df_claim['pgpub_id'].astype(str)

        result_dict = match_patent_id(df_claim, pgpub_ids)
        print(f"Number of matched pgpub_ids in current file: {len(result_dict)}")

        d_claims.update(result_dict)
        print(f"Total number of pgpub_ids with claims so far: {len(d_claims)}")

    # Concatenate claims for each pgpub_id
    concatenated_claims_dict = {}

    for pgpub_id, claims in d_claims.items():
        # Sort the claims based on their claim number
        sorted_claims = sorted(claims.items())

        # Concatenate the claim texts in order
        concatenated_claims = ' '.join(f"{claim_text}" for claim_number, claim_text in sorted_claims)

        # Calculate the length of the concatenated string
        concatenated_length = len(concatenated_claims)

        # Store the concatenated string and its length in the new dictionary
        concatenated_claims_dict[pgpub_id] = {'claims_text': concatenated_claims, 'claims_length': concatenated_length}

    # Convert the dictionary to a DataFrame
    concatenated_claims_df = pd.DataFrame.from_dict(concatenated_claims_dict, orient='index').reset_index()
    concatenated_claims_df.columns = ['pgpub_id', 'claims_text', 'claims_length']

    # Ensure 'pgpub_id' is a string in both DataFrames
    dataset_df['pgpub_id'] = dataset_df['pgpub_id'].astype(str)
    concatenated_claims_df['pgpub_id'] = concatenated_claims_df['pgpub_id'].astype(str)

    # Merge the new DataFrame with your existing one
    merged_df = pd.merge(dataset_df, concatenated_claims_df, on='pgpub_id', how='left')

    # Drop rows for which no claims were found
    merged_df = merged_df.dropna(subset=['claims_text'])

    # Calculate statistics
    min_value = merged_df['claims_length'].min()
    max_value = merged_df['claims_length'].max()
    mean_value = merged_df['claims_length'].mean()

    # Print the results
    print(f"Statistics for {dataset_name}:")
    print(f"Minimum claims length: {min_value}")
    print(f"Maximum claims length: {max_value}")
    print(f"Mean claims length: {mean_value}")

    return merged_df

print("Processing claims for pre-2010 data...")
pre_year_with_claims = process_claims_for_dataset(pre_year_df, 'Pre-2010 Data')

print("Processing claims for post-2010 data...")
post_year_with_claims = process_claims_for_dataset(post_year_df, 'Post-2010 Data')

# Paths to save the new data files
output_dir = f"{PROJ_DIR}/data/processed_data/claims/{SECTION}-{YEAR_THRESHOLD}"
os.makedirs(output_dir, exist_ok=True)

pre_year_claims_file = os.path.join(output_dir, f'pre_{YEAR_THRESHOLD}_claims_text.csv')
post_year_claims_file = os.path.join(output_dir, f'post_{YEAR_THRESHOLD}_claims_text.csv')

# Save the DataFrames
pre_year_with_claims.to_csv(pre_year_claims_file, index=False)
print(f"Pre-2010 data with claims saved to {pre_year_claims_file}")

post_year_with_claims.to_csv(post_year_claims_file, index=False)
print(f"Post-2010 data with claims saved to {post_year_claims_file}")