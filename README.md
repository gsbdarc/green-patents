# BERT Patent Training Project Pipeline
## Overview and instructions for the project
This repository contains scripts for training BERT models on patent claims and details. The project is structured to include data preprocessing, tokenization, model training, prediction, and evaluation steps.

### Setup
1. Navigate to `src` folder:

```
cd src
```

2. Create a virtual environment:

```bash
/usr/bin/python3 -m venv venv
```

3. Activate the environment:

```
source venv/bin/activate
```

4. Install required packages:

```
pip install -r requirements.txt
```

5. Create a Jupyter kernel for this environment:

```
python -m ipykernel install --user --name=patents
```

### Edit `config.yml`
Here is where the user picks the section to filter on from 8 different CPC sections and what year to use as a split. Only data before the split year will be used in training but all years will be predicted on. 

Set the following variables to what section and year to use for data processing:
```
section: 'F'
year: 2010
```

Pick what labels you want to use as ground truth. There are different patent classification schemes such as WIPO and CPC. In the config file, set the path to the label file that corresponds to the classification scheme of choice.

```
label_file: '/zfs/projects/faculty/jinhwan-green-patents/data/labels/all_patents_CPC_groups.csv'
```

### Data Preprocessing
1. Navigate to the data preparation folder:
```
cd data_prep
``` 

2. Run the data preparation script (this requires that the `venv` from Setup step is active):
```
python 0_data_prep.py
```

3. Submit Slurm jobs to extract and save text data:

- Save claims text:
  ```
  sbatch 1_save_claims_text.slurm
  ```  

- Save details text:
  ```
  sbatch 1_save_details_text.slurm
  ```

4. Submit Slurm jobs to tokenize the data:

- Tokenize claims:
  ```
  sbatch 2_tokenize_claims.slurm
  ```

- Tokenize details:
  ```
  sbatch 2_tokenize_details.slurm
  ```

5. Generate train/validation/test data:

```
python 3_make_train_data.py
```

### Model Training

Navigate to the model training folder and submit the training scripts:
``` 
cd model_training
```

Train on Claims data:
```
sbatch claims/train.slurm
```

Train on Details data:
```
sbatch details/train.slurm
```

Training will be run for `max_epochs` (set to 3 as default) epochs. Since we can only request one day on `gpu` partition, the script is written to pick up where it left off if the `max_epochs` have not been completed.

### Label Prediction
Navigate to the model inference folder and submit the prediction scripts:

```
cd inference
```

Predict on chunks of Claims data:
```
sbatch claims/predict_on_chunks.slurm
```

Predict on chunks of Details data (tokenized details are big so we break the prediciton into two concurrent script that can be run in parallel) to predict on all document chunks:
```
sbatch details/predict_pre_year.slurm
sbatch details/predict_post_year.slurm
```

### Prediction Evaluation

Lastly, convert chunk probabilities into document probabilities:

- Run `src/inference/claims/combine_chunks_per_doc.ipynb`
- Run `src/inference/details/combine_chunks_per_doc.ipynb`

### Results
After the pipeline is run, the final document probabilities (of "green" label) are written to:

- Claims model predictions:
  ```
  predictions/claims-model/claims_model_doc_preds.csv
  ```

- Details model predictions:
  ```
  predictions/details-model/details_model_doc_preds.csv
  ```