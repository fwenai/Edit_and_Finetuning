# On The Retention Of Edited Knowledge In Fine-tuned Language Models
This repository provides an implementation of the paper On The Retetion OF Edited Knowledge In Finetuned Language Models

## Running code
Environment versions: please see environment.yml

## Main Experiment

### Environmental setup
    ```
    conda edit_and_finetuning_env create -f environment.yml
    ```
### Rome edit for the model GPT-XL, 
for each edited knowledge, 1. create a train dataset that is irrelevant to the edited knowledge 2. sample a validation dataset for stopping criteria 3. perform knowledge edition 4. perform downstream finetuning 5. compute the edited knowledge retention rate 6. compute the intrinsic knowledge retention rate
    ```
    sh run_all.sh
    ```

