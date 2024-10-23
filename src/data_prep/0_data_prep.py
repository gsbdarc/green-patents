# Import necessary libraries
import os
import pandas as pd
import yaml

PROJ_DIR = '/zfs/projects/faculty/jinhwan-green-patents'

# Step 1: Load Configuration
# Assume the YAML file is named 'config.yaml'
with open(f'{PROJ_DIR}/config.yaml', 'r') as file:
    config = yaml.safe_load(file)

# Extract configuration parameters
SECTION = config['section']                 # e.g., 'F'
YEAR_THRESHOLD = config['year']             # e.g., 2010
LABEL_FILE = config['label_file']           # e.g., 'all_patents_CPC_groups.csv'
PRE_2010_DATA_FILE = config['pre_2010_data_file']   # e.g., 'PreGrant_patent_2001_2010.dta'
POST_2010_DATA_FILE = config['post_2010_data_file'] # e.g., 'PreGrant_patent_post_2010.dta'
SECTIONS_FILE = config['sections_file']     # e.g., 'sections-all-pg_pubs.csv'

# Step 2: Set Up Output Directory
output_dir = f"{PROJ_DIR}/data/data_prep/{SECTION}-{YEAR_THRESHOLD}"
os.makedirs(output_dir, exist_ok=True)

# Check if the data already exists
pre_year_file = os.path.join(output_dir, f'pre_{YEAR_THRESHOLD}_pgpub.csv')
post_year_file = os.path.join(output_dir, f'post_{YEAR_THRESHOLD}_pgpub.csv')

data_files_exist = all([
    os.path.exists(pre_year_file),
    os.path.exists(post_year_file)
])

if data_files_exist:
    print(f"Data for section '{SECTION}' and year '{YEAR_THRESHOLD}' already exists in '{output_dir}'.")
    print("Skipping preprocessing.")
else:
    print(f"Data for section '{SECTION}' and year '{YEAR_THRESHOLD}' does not exist.")
    print("Running preprocessing...")

    # Step 3: Load Data Files
    # Load pre-2010 and post-2010 data files
    pre_2010_df = pd.read_stata(PRE_2010_DATA_FILE)
    post_2010_df = pd.read_stata(POST_2010_DATA_FILE)

    # Step 4: Merge the DataFrames
    # We only keep 'pgpub_id' and 'year' columns
    pg_df = pd.concat(
        [pre_2010_df[['pgpub_id', 'year']], post_2010_df[['pgpub_id', 'year']]],
        ignore_index=True
    )
    print(f'pg_df.shape: {pg_df.shape}')

    # First, cast 'pgpub_id' as int (ids have .0 at the end)
    pg_df['pgpub_id'] = pg_df['pgpub_id'].astype(int)
    # Ensure 'pgpub_id' is a string
    pg_df['pgpub_id'] = pg_df['pgpub_id'].astype(str)

    # Step 5: Load Sections File
    sections_data = pd.read_csv(SECTIONS_FILE)
    # Ensure 'pgpub_id' is a string
    sections_data['pgpub_id'] = sections_data['pgpub_id'].astype(str)

    # Step 6: Load Labels File
    labels_data = pd.read_csv(LABEL_FILE)
    # Ensure 'pgpub_id' is a string
    labels_data['pgpub_id'] = labels_data['pgpub_id'].astype(str)

    # Step 7: Merge Data
    # Merge pg_df with sections_data on 'pgpub_id'
    merged_data = pd.merge(pg_df, sections_data, on='pgpub_id', how='left')

    # Merge the result with labels_data on 'pgpub_id'
    merged_data = pd.merge(merged_data, labels_data, on='pgpub_id', how='left')

    # Step 8: Drop Unnecessary Columns
    # Keep only 'pgpub_id', 'year', 'section', and 'label'
    columns_to_keep = ['pgpub_id', 'year', 'section', 'label']
    merged_data = merged_data[columns_to_keep]

    # Drop rows with missing 'section' or 'label'
    merged_data.dropna(subset=['section', 'label'], inplace=True)

    # Step 9: Filter by Section
    filtered_data = merged_data[merged_data['section'] == SECTION]

    # Step 10: Split Data Based on Year
    # Convert 'year' to numeric if necessary
    filtered_data['year'] = pd.to_numeric(filtered_data['year'], errors='coerce')

    # Drop rows with NaN years if any
    filtered_data = filtered_data.dropna(subset=['year'])

    print(f'filtered_data.shape: {filtered_data.shape}')

    # Split into pre-year and post-year data
    pre_year_data = filtered_data[filtered_data['year'] <= YEAR_THRESHOLD]
    post_year_data = filtered_data[filtered_data['year'] > YEAR_THRESHOLD]

    print(f'pre_year_data.shape: {pre_year_data.shape}')
    print(f'post_year_data.shape: {post_year_data.shape}')

    # Save pre-year and post-year data
    pre_year_data.to_csv(pre_year_file, index=False)
    post_year_data.to_csv(post_year_file, index=False)

    print("Data preprocessing complete. Datasets saved:")
    print(f"Pre-year data shape: {pre_year_data.shape} saved to {pre_year_file}")
    print(f"Post-year data shape: {post_year_data.shape} saved to {post_year_file}")
