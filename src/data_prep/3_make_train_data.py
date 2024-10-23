import os
import pandas as pd
from sklearn.model_selection import train_test_split
import yaml

# Configurable paths
PROJ_DIR = '/zfs/projects/faculty/jinhwan-green-patents'

# Step 1: Load Configuration
with open(f'{PROJ_DIR}/config.yaml', 'r') as file:
    config = yaml.safe_load(file)

# Extract configuration parameters
SECTION = config['section']  # e.g., 'F'
YEAR_THRESHOLD = config['year']  # e.g., 2010

# Input paths for the tokenized claims and details data
tokenized_claims_file = f"{PROJ_DIR}/data/processed_data/tokenized_claims/{SECTION}-{YEAR_THRESHOLD}/pre_{YEAR_THRESHOLD}_claims_tokenized.csv"
tokenized_details_file = f"{PROJ_DIR}/data/processed_data/tokenized_details/{SECTION}-{YEAR_THRESHOLD}/pre_{YEAR_THRESHOLD}_details_tokenized.csv"

# Output directory for split datasets
output_dir_claims = f"{PROJ_DIR}/data/processed_data/claims/{SECTION}-{YEAR_THRESHOLD}"
output_dir_details = f"{PROJ_DIR}/data/processed_data/details/{SECTION}-{YEAR_THRESHOLD}"
os.makedirs(output_dir_claims, exist_ok=True)
os.makedirs(output_dir_details, exist_ok=True)

# Step 1: Load tokenized claims data
print(f"Loading tokenized claims data from {tokenized_claims_file}")
claims_df = pd.read_csv(tokenized_claims_file)
print(f"Claims data shape: {claims_df.shape}")

# Header claims_df
# pgpub_id,label,year,tokenized_chunk,chunk_id

# Step 2: Get unique pgpub_ids
unique_ids_with_labels = claims_df[['pgpub_id', 'label']].drop_duplicates()

# Perform stratified split based on label
train_ids, temp_ids = train_test_split(
    unique_ids_with_labels['pgpub_id'], 
    test_size=0.3, 
    random_state=42, 
    stratify=unique_ids_with_labels['label']
)  # 70% train, 30% temp (for val + test)

# Perform stratified split for validation and test from the remaining data
temp_ids_with_labels = unique_ids_with_labels[unique_ids_with_labels['pgpub_id'].isin(temp_ids)]
val_ids, test_ids = train_test_split(
    temp_ids_with_labels['pgpub_id'], 
    test_size=0.5, 
    random_state=42, 
    stratify=temp_ids_with_labels['label']
)  # 15% val, 15% test

# Step 2: Filter the original DataFrame based on the split pgpub_ids
train_claims_df = claims_df[claims_df['pgpub_id'].isin(train_ids)]
val_claims_df = claims_df[claims_df['pgpub_id'].isin(val_ids)]
test_claims_df = claims_df[claims_df['pgpub_id'].isin(test_ids)]

# Print sizes of each split
print(f"Training set size: {len(train_claims_df)}")
print(f"Validation set size: {len(val_claims_df)}")
print(f"Test set size: {len(test_claims_df)}")

# Optional: Verify the pgpub_id counts
print(train_claims_df['pgpub_id'].nunique(), val_claims_df['pgpub_id'].nunique(), test_claims_df['pgpub_id'].nunique())

# Print the sizes of each claims split
print(f"Train claims shape: {train_claims_df.shape}")
print(f"Validation claims shape: {val_claims_df.shape}")
print(f"Test claims shape: {test_claims_df.shape}")

# Step 3: Load tokenized details data
print(f"Loading tokenized details data from {tokenized_details_file}")
details_df = pd.read_csv(tokenized_details_file)
print(f"Details data shape: {details_df.shape}")

# Step 4: Filter details data using the same pgpub_ids from claims split
train_details_df = details_df[details_df['pgpub_id'].isin(train_ids)]
val_details_df = details_df[details_df['pgpub_id'].isin(val_ids)]
test_details_df = details_df[details_df['pgpub_id'].isin(test_ids)]

# Print the sizes of each details split
print(f"Train details shape: {train_details_df.shape}")
print(f"Validation details shape: {val_details_df.shape}")
print(f"Test details shape: {test_details_df.shape}")

# Step 5: Save the datasets
# Save the claims splits
train_claims_file = os.path.join(output_dir_claims, f"train_{YEAR_THRESHOLD}_claims_tokenized.csv")
val_claims_file = os.path.join(output_dir_claims, f"val_{YEAR_THRESHOLD}_claims_tokenized.csv")
test_claims_file = os.path.join(output_dir_claims, f"test_{YEAR_THRESHOLD}_claims_tokenized.csv")

train_claims_df.to_csv(train_claims_file, index=False)
val_claims_df.to_csv(val_claims_file, index=False)
test_claims_df.to_csv(test_claims_file, index=False)

print(f"Train claims saved to {train_claims_file}")
print(f"Validation claims saved to {val_claims_file}")
print(f"Test claims saved to {test_claims_file}")

# Save the details splits
train_details_file = os.path.join(output_dir_details, f"train_{YEAR_THRESHOLD}_details_tokenized.csv")
val_details_file = os.path.join(output_dir_details, f"val_{YEAR_THRESHOLD}_details_tokenized.csv")
test_details_file = os.path.join(output_dir_details, f"test_{YEAR_THRESHOLD}_details_tokenized.csv")

train_details_df.to_csv(train_details_file, index=False)
val_details_df.to_csv(val_details_file, index=False)
test_details_df.to_csv(test_details_file, index=False)

print(f"Train details saved to {train_details_file}")
print(f"Validation details saved to {val_details_file}")
print(f"Test details saved to {test_details_file}")

print(f"Splitting complete for both claims and details.")
