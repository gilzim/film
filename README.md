# FiBN: Visual Reasoning with a General Conditioning Layer and Conditional Batch Normalization

### Folder Structure

This code is a fork from the code for "FiLM: Visual Reasoning with a General Conditioning Layer" available [here](https://github.com/ethanjperez/film).

the src folder contains 3 folders: vr, scripts and img.

the vr folder includes .py files for preprocessing data, a utils.py file for loading models from checkpoints
and the models package which contatins the implementations of the different layers and models.

the scripts folder includes .py files for preprocessing the data, .py files for training and running models 
and  the train folder which contains .sh scripts for training FiLM and FiBN models.

the img folder includes example pictures of the CLEVR dataset and the stats folder which contains gammas and betas distributions of the FiBN model.

### Important Notes
- The code can only run on the Linux OS
- All bash scripts and commands must be executed from the src folder

### Setup
First, create an empty conda enviorment and run the following command: 
```bash
pip install -r requirements.txt
```

Second, follow the CLEVR data preprocessing [instructions](https://github.com/facebookresearch/clevr-iep/blob/master/TRAINING.md#preprocessing-clevr).

### Training
The below script has the hyperparameters and settings to reproduce FiBN CLEVR results:
```bash
sh scripts/train/fibn.sh
```
Training a FiBN CLEVR model should take ~20 hours on an average GPU

### Running models

There is an interactive command line tool for use with the below command/script.
```bash
python run_model.py --program_generator <FiLM Generator filepath> --execution_engine <FiLMed Network filepath>
```
When FiLM Generator filepath and FiLMed Network filepath are the same.

By default, the command runs on [this CLEVR image](https://github.com/gilzim/film/blob/CBN_layers/img/CLEVR_val_000017.png), but you may modify which image to use via command line flag to test on any CLEVR image.

We added the script run_model_fibn.sh which runs on a batch of 3000 samples. It returns the accuracy of the model
and saves different gammas & bettas distributions in img/stats
