# ML_Project_Template
Building a machine learning project using IPython Notebook and Python Virtual Environment 

## Install Jupiter NoteBook 
```
pip3 install --upgrade pip

pip3 install jupyter
```
## Pip3 Installation Problem Solution
https://stackoverflow.com/questions/49836676/python-pip3-cannot-import-name-main-error-after-upgrading-pip?utm_medium=organic&utm_source=google_rich_qa&utm_campaign=google_rich_qa

## Setting Python  RunTime via virtual environment 

Step 1: Go go the project directory , then 1) create a directory for virtual enviroment , add python libary to be installed to the requirements.txt file and then install all the requirements 
```
mkdir project_dependencies
virtualenv -p /usr/local/bin/python3.6 project_dependencies/
source project_dependencies/bin/activate
nano requirements.txt
pip install -r requirements.txt
```
Step 2: Select the kernal depedency from virtual environment 

```
pip install ipykernel
ipython kernel install --user --name=project_dependencies

```
Step 3: Run the jupiter Notebook by the follwing command 

```
jupyter notebook

```

## Experiment

### Data Collection: 
 Data is collected from http://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data

### Methods 
Deep Learning - Recurrent Neural Network 

Machine Learning - Decision Tree, Support Vector Machine, GradientBoosting


### Run 
- Go the models/machine_learning/decision_tree 
- Run the main.py file 
- Result will be saved in results folder
### Result 
Every Run of a script gives a new result file  which contains the information like below 

```
{
    "accuracy": 0.9649122807017544,
    "confusion_matrix": [
        [
            134,
            4
        ],
        [
            4,
            86
        ]
    ],
    "hyper_parameters": {
        "learning_rate": 0.1
    },
    "labels": [
        "Malignant",
        "benign"
    ],
    "precision": 0.9306302794022092,
    "recall": 0.9555555555555556
}
```






