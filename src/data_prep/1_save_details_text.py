import pandas as pd
import glob
import json
import os
import yaml

# Configurable paths
PROJ_DIR = '/zfs/projects/faculty/jinhwan-green-patents'
DETAILS_DIR = f'{PROJ_DIR}/data/raw_data/pg_details'

# Step 1: Load Configuration
with open(f'{PROJ_DIR}/config.yaml', 'r') as file:
    config = yaml.safe_load(file)

# Extract configuration parameters
SECTION = config['section']                 # e.g., 'F'
YEAR_THRESHOLD = config['year']             # e.g., 2010

# Load section-year data to process
input_dir = f"{PROJ_DIR}/data/data_prep/{SECTION}-{YEAR_THRESHOLD}"
pre_year_file = os.path.join(input_dir, f'pre_{YEAR_THRESHOLD}_pgpub.csv')
post_year_file = os.path.join(input_dir, f'post_{YEAR_THRESHOLD}_pgpub.csv')

# Output directory
output_dir = f"{PROJ_DIR}/data/processed_data/details/{SECTION}-{YEAR_THRESHOLD}"
os.makedirs(output_dir, exist_ok=True)

# Load the pre-year and post-year data
pre_year_df = pd.read_csv(pre_year_file, dtype={'pgpub_id': str})
post_year_df = pd.read_csv(post_year_file, dtype={'pgpub_id': str})

def process_details_for_dataset(dataset_df, dataset_name):
    """
    Processes details for the given dataset DataFrame and returns the DataFrame with description text added.
    """
    pgpub_ids = set(dataset_df['pgpub_id'])
    print(f"Number of pgpub_ids in {dataset_name}: {len(pgpub_ids)}")

    # Initialize a dictionary to hold description texts
    d_descriptions = {}

    # Get all details files
    details_files = glob.glob(f'{DETAILS_DIR}/*.tsv')
    print(f'Number of details files: {len(details_files)}')

    for f_i, f in enumerate(details_files):
        print(f"Processing file: {f}")
        df_details = pd.read_csv(f, delimiter='\t', low_memory=False)

        # Ensure 'pgpub_id' is a string
        df_details['pgpub_id'] = df_details['pgpub_id'].astype(str)

        # Filter df_details based on pgpub_ids
        df_tmp = df_details[df_details['pgpub_id'].isin(pgpub_ids)]
        print(f"Number of matched pgpub_ids in current file: {df_tmp['pgpub_id'].nunique()}")

        # If any duplicates, keep the first occurrence
        df_tmp = df_tmp.drop_duplicates(subset='pgpub_id', keep='first')

        # Update the descriptions dictionary
        descriptions = df_tmp.set_index('pgpub_id')['description_text'].to_dict()
        d_descriptions.update(descriptions)
        print(f"Total number of pgpub_ids with descriptions so far: {len(d_descriptions)}")

    # Convert the dictionary to a DataFrame
    descriptions_df = pd.DataFrame.from_dict(d_descriptions, orient='index').reset_index()
    descriptions_df.columns = ['pgpub_id', 'description_text']

    # Merge the new DataFrame with your existing one
    merged_df = pd.merge(dataset_df, descriptions_df, on='pgpub_id', how='left')

    # Drop rows for which no description was found
    merged_df = merged_df.dropna(subset=['description_text'])

    # Optionally, process the description_text (e.g., JSON encode)
    # merged_df['description_text'] = merged_df['description_text'].apply(lambda x: json.dumps(x))

    # Calculate statistics
    min_length = merged_df['description_text'].str.len().min()
    max_length = merged_df['description_text'].str.len().max()
    mean_length = merged_df['description_text'].str.len().mean()

    print(f"Statistics for {dataset_name}:")
    print(f"Minimum description length: {min_length}")
    print(f"Maximum description length: {max_length}")
    print(f"Mean description length: {mean_length}")

    return merged_df

print("Processing details for pre-2010 data...")
pre_year_with_details = process_details_for_dataset(pre_year_df, 'Pre-2010 Data')

print("Processing details for post-2010 data...")
post_year_with_details = process_details_for_dataset(post_year_df, 'Post-2010 Data')

# Paths to save the new data files
pre_year_details_file = os.path.join(output_dir, f'pre_{YEAR_THRESHOLD}_details_text.csv')
post_year_details_file = os.path.join(output_dir, f'post_{YEAR_THRESHOLD}_details_text.csv')

# Save the DataFrames
pre_year_with_details.to_csv(pre_year_details_file, index=False)
print(f"Pre-2010 data with descriptions saved to {pre_year_details_file}")

post_year_with_details.to_csv(post_year_details_file, index=False)
print(f"Post-2010 data with descriptions saved to {post_year_details_file}")
