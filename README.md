# MnistOpe

The goal of this project is to create an operation network that work on Mnist Dataset.

# Project Structure
```
| - README.md
| - LICENSE
| - .gitignore
|
| - classifier   (It's not my work : Source from [https://github.com/pytorch/examples/blob/00ea159a99f5cb3f3301a9bf0baa1a5089c7e217/mnist/main.py](Pytorch example) )
|   | - main.py       (Run it with ```python main.py --help```)
|   | - dataloader.py (Download data inside the data directory)
|   | - models.py     (Simple LeNet Classifier for MNIST)
|   | - train.py      (Train function called in main.py) 
|   | - test.py       (Test function called in main.py)
|
| - data
|   | - MNIST     (Downloaded data)
|   |   | - raw
|   |   |   |  - train-images-idx3-ubyte.gz
|   |   |   |  - train-labels-idx1-ubyte.gz
|   |   |   |  - t10k-images-idx3-ubyte.gz
|   |   |   |  - t10k-labels-idx1-ubyte.gz
|
| - sum_classifier (This is my work : Implement a simple LeNet + Concatenation Sum Classifier for MNIST)
|   | - main.py       (Run it with ```python main.py --help```)
|   | - dataloader.py (Download data inside the data directory + Create a custom dataloader (2 images + label the sum))
|   | - models.py     (Custom SumNet classifier (Takes 2 images + return the sum))
|   | - train.py      (Train function called in main.py)
|   | - test.py       (Test function called in main.py)
|
| - ops_classifier (This is also my work : Implement a simple LeNet + Concatenation Ops Classifier for MNIST)
|   | - main.py       (Run it with ```python main.py --help```)
|   | - dataloader.py (Download data inside the data directory + Create a custom dataloader (2 images + 1 operator + label the sum))
|   | - models.py     (Custom OpNet classifier (Takes 2 images and an operator + return operator(im1, im2)))
|   | - train.py      (Train function called in main.py)
|   | - test.py       (Test function called in main.py)
|
| - weights
|   | - classifier
|   |   | - weights.pth   (Weights for the basic classifier)
|   | - sum_classifier
|   |   | - weights.pth   (Weights for the sum classifier)
|   | - ops_classifier
|   |   | - weights.pth   (Weights for the ops classifier)
```
