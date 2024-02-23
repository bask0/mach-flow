### Deep learning project seed
Use this seed to start new deep learning / ML projects.

- Built in setup.py
- Built in requirements
- Examples with MNIST
- Badges
- Bibtex

#### Goals  
The goal of this seed is to structure ML paper-code the same so that work can easily be extended and replicated.   

### DELETE EVERYTHING ABOVE FOR YOUR PROJECT  
 
---

<div align="center">    
 
# CH-RUN: A data-driven spatially contiguous runoff monitoring product for Switzerland   

<!--Change batch name and link to paper -->
[![Paper](https://img.shields.io/badge/HESS-in%20prep.-blue.svg)](https://www.hydrology-and-earth-system-sciences.net/)  


</div>
 
## Description   
What it does   

## How to run   
First, install dependencies   
```bash
# clone project   
git clone https://github.com/YourGithubName/deep-learning-project-template

# install project   
cd deep-learning-project-template 
pip install -e .   
pip install -r requirements.txt
 ```   
 Next, navigate to any file and run it.   
 ```bash
# module folder
cd project

# run module (example: mnist as your main contribution)   
python lit_classifier_main.py    
```

## Imports
This project is setup as a package which means you can now easily import any file into any other file like so:
```python
from project.datasets.mnist import mnist
from project.lit_classifier_main import LitClassifier
from lightning.pytorch import Trainer

# model
model = LitClassifier()

# data
train, val, test = mnist()

# train
trainer = Trainer()
trainer.fit(model, train, val)

# test using the best model!
trainer.test(test_dataloaders=test)
```

### Citation   
```
@article{kraft_chrun_2024,
  title={CH-RUN: A data-driven spatially contiguous runoff monitoring product for Switzerland},
  author={B. Kraft, W. Aeberhard, M. Schirmer, M. Zappa, S. I. Seneviratne and L. Gudmundsson},
  journal={HESS},
  year={in prep.}
}
```
