# CSS_Sarcasm
Project for CS6471 Spring 2022 Georgia Tech

## PROJECT NAME
Analyzing Context and User Information in Online Sarcasm Detection


## TEAM

### TEAM NAME
Team Metaverse

### TEAM MEMBERS
1. Tusheet Sidharth Goli
2. Mohan Dodda


## CODE

### DATASET
Our dataset was too large to host on GitHub, so we have linked the dataset we used on Kaggle.

Dataset - https://www.kaggle.com/datasets/mohandodda/wgca-user-embeddings


### DIRECTORY STRUCTURE
1. The documents folder contains the deliverables like the project proposal, midway report, and final report. It also contains the midway presentation and final poster.
2. The code folder contains all the python files that can be run to obtain our results. This is primarily what we used for our research and analysis during this project.
4. The root folder contains the README.md file.
5. The .idea folder can be ignored.

### INSTALLATION INSTRUCTIONS
1. To be able to run the code from the code folder, create a directory called data in the root directory and download the datasets individually from the link above.
2. You will need the appropriate packages. To do this use the environment.yml file. Make a conda environment with conda env create -f environment. yml and then activate the enviroment with conda activate cascade

### RUNNING CODE
All code are in jupyter notebooks which might be a bit more intuitive to use. If you want to just use python you can as well. Here the appropriate use cases and commands.
First cd into the code folder!

To do statistical analysis on the contextual features on the code go to `understanding-sarcasm-tusheet.ipynb`

To run the feature based model and generate feature importances run 
`python featureimportance.py`

These next files are for training the BERT, BERT+CASCADE, and Logistic Regression models. for these files if you want to change datasets you will have to change a variable called datasettype in the corresponding python file. 

For parent dataset set datasettype to "parent"

For response dataset set datasettype to "response"

For parent+response dataset set datasettype to "parentresponse"

To run fine-tune a BERT model on the different datasets 
run `python BERTTraining.py`. This model allows all 3 datasets.

To run fine-tune a BERT+CASCADE model on parent or parent+response dataset.
  `python logisicreg-bert-cascade-parent-only-combined.py`.  This file also does statistical significance and error modeling. 
  
To run fine-tune a BERT+CASCADE model python. This model allows the response dataset.
run `python LogisticReg-and-BERT+CASCADEresponse.py`.  This file also does statistical significance and error modeling.
