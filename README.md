# FiBN: Visual Reasoning with a General Conditioning Layer and Conditional Batch Normalization

### Fodler Structure

This code is a fork from the code for "FiLM: Visual Reasoning with a General Conditioning Layer" available [here](https://github.com/ethanjperez/film).

the src folder contains 3 folders: vr, scripts and img.

the vr folder includes .py files for preprocessing data, a utils.py file for loading models from checkpoints
and the models package which contatins the implementations of the different layers and models.

the scripts folder includes .py files for preprocessing the data, .py files for training and running models 
and  the train folder which contains .sh scripts for training film and fibn models.

the img folder includes example pictures of the CLEVR dataset and the stats folder which contains gammas and betas distributions of the fibn model.

### Setup
Becuase we essentialy use the FiLM model, the setup instructions for the FiBN model are the same as FiLM.
from FiLM's README.md:

First, follow the virtual environment setup [instructions](https://github.com/facebookresearch/clevr-iep#setup).

Second, follow the CLEVR data preprocessing [instructions](https://github.com/facebookresearch/clevr-iep/blob/master/TRAINING.md#preprocessing-clevr).

### Trainning
The below script has the hyperparameters and settings to reproduce FiBN CLEVR results:
```bash
sh scripts/train/fibn.sh
```
For CLEVR-Humans, data preprocessing instructions are [here](https://github.com/facebookresearch/clevr-iep/blob/master/TRAINING.md#preprocessing-clevr-humans).
The below script has the hyperparameters and settings to reproduce FiBN CLEVR-Humans results:
```bash
sh scripts/train/fibn_humans.sh
```
Training a FiBN CLEVR model should take ~20 hours on an avarage GPU

### Running models

There is an interactive command line tool for use with the below command/script.
```bash
python run_model.py --program_generator <FiLM Generator filepath> --execution_engine <FiLMed Network filepath>
```
When FiLM Generator filepath and FiLMed Network filepath are the same.

By default, the command runs on [this CLEVR image](img/CLEVR_val_000017.png), but you may modify which image to use via command line flag to test on any CLEVR image.

CLEVR vocab is enforced by default, but for CLEVR-Humans models, for example, you may append the command line flag option 
'--enforce_clevr_vocab 0' to ask any string of characters you please.

We added the script run_model_fibn.sh which runs on a batch of 3000 samples. It returns the accuracy of the model
and saves different gammas & bettas distributions in img/stats
