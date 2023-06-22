# Session 7 Assignment
Model to detect handwritten digits, trained on MNIST dataset of 60,000 images.

**Goal is to create a model with**
- 99.4% validation accuracy with consistency
- Less than 8k Parameters
- Less than 15 Epochs

## Model_1.py
<table>
        <tr>
                <th>Target</th>
                <th>Result</th>
                <th>Analysis</th>
        </tr>
        <tr>
                <td>
                        <ol>
                        <li>Building data loaders, test & train data sets and train & test loop</li>
                        <li>Also, setting basic skeleton with working model (without shape, size errors in model) </li>
                        <li>Working model should be able to reach 98-99% accuracy on the dataset with the skeleton model</li>
                        </ol>
                </td>
                <td>
                        <ol>
                        <li>390K+ parameters </li>
                        <li>Best training accuracy - 99.94% </li>
                        <li>Best test accuracy - 99.41% </li>
                        </ol>
                </td>
                <td>
                        <ol>
                        <li>Too many parameters, need lighter model </li>
                        <li>Overfitting </li>
                        </ol>
                </td>
        </tr>
        <tr>
                <td>
                        Building a lighter model with params under 30k
                </td>
                <td>
                        <ol>
                                <li>~26K parameters</li>
                                <li>Best training accuracy - 99.84</li>
                                <li>Best test accuracy - 99.23%</li>
                        </ol>
                </td>
                <td>
                        <ol>
                                <li>Skeleton working, need to further reduce the params</li>
                                <li>Overfitting</li>
                        </ol>
                </td>
        </tr>
</table>

## Model_2.py
<table>
        <tr>
                <th>Target</th>
                <th>Result</th>
                <th>Analysis</th>
        </tr>
        <tr>
                <td>
                        Building a lighter model with params under 8k
                </td>
                <td>
                        <ol>
                        <li>6.7k parameters</li>
                        <li>Best training accuracy - 99.45 (20th epoch)</li>
                        <li>Best test accuracy - 98.97% (13th epoch)</li>
                        </ol>
                </td>
                <td>
                        <ol>
                        <li>Good model and can be pushed further</li>
                        <li>Overfitting </li>
                        </ol>
                </td>
        </tr>
        <tr>
                <td>
                        Add normalisation, BatchNorm to push model efficiency
                </td>
                <td>
                        <ol>
                                <li>~6.9k params</li>
                                <li>Best training accuracy - 99.74%</li>
                                <li>Best test accuracy - 99.21%</li>
                        </ol>
                </td>
                <td>
                        <ol>
                                <li>Still there's overfitting</li>
                                <li>Model efficiency can't be pushed further</li>
                        </ol>
                </td>
        </tr>
         <tr>
                <td>
                        Add regularization (Dropout) to get rid of overfitting
                </td>
                <td>
                        <ol>
                                <li>Best training accuracy - 98.90 (19th epoch)/li>
                                <li>Best test accuracy - 99.30% (18th epoch)</li>
                        </ol>
                </td>
                <td>
                        <ol>
                                <li>Underfitting but that's because of regularisation, Good</li>
                                <li>Model can't be pushed further with current capacity</li>
                        </ol>
                </td>
        </tr>
</table>

## Model_3.py
<table>
        <tr>
                <th>Target</th>
                <th>Result</th>
                <th>Analysis</th>
        </tr>
        <tr>
                <td>
                        Add GAP & remove last layer
                </td>
                <td>
                        <ol>
                        <li>4.5k parameters</li>
                        <li>Best training accuracy - 98.74 (20th epoch)</li>
                        <li>Best test accuracy - 99.28% (18th Epoch)</li>
                        </ol>
                </td>
                <td>
                        <ol>
                        <li>Need to add capacity to reach 99.4 goal</li>
                        <li>No overfitting </li>
                        </ol>
                </td>
        </tr>
        <tr>
                <td>
                        Add FC post GAP layer and see
                </td>
                <td>
                        <ol>
                                <li>Model parameters - 6.5k</li>
                                <li>Best training accuracy - 99.11% (20th epoch)</li>
                                <li>Best test accuracy - 99.36% (20th epoch)</li>
                        </ol>
                </td>
                <td>
                        <ol>
                                <li>Underfitting but it's fine. </li>
                                <li>Need to playaround with image transforms to make training difficult</li>
                        </ol>
                </td>
        </tr>
         <tr>
                <td>
                        <ol>
                                <li>Add transformations to input dataset</li>
                                <li>Need to add rotation between (-10) to (10) degree</li>
                        </ol>
                </td>
                <td>
                        <ol>
                                <li>Best training accuracy - 98.36 (17th epoch)/li>
                                <li>Best test accuracy - 99.34% (12th epoch)</li>
                        </ol>
                </td>
                <td>
                        <ol>
                                <li>Training is difficult enough</li>
                                <li>Need to reduce epochs using right LR strategy</li>
                        </ol>
                </td>
        </tr>
        <tr>
                <td>
                        Achieve 99.4% accuracy within 15 epochs. Trying out ReduceLROnPlateau (Learning rate 0.1, patience=2, threshold = 0.001)
                        </ol>
                </td>
                <td>
                        <ol>
                                <li>Best training accurancy - 98.73 (14th epoch)/li>
                                <li>Best test accuracy - 99.41 (12th epoch)</li>
                        </ol>
                </td>
                <td>
                        Learning rate & batch size directly affects number of epochs
                </td>
        </tr>
</table>

Below is the final model (Model_3) summary -
```
----------------------------------------------------------------

        Layer (type)               Output Shape         Param #

================================================================

            Conv2d-1            [-1, 8, 28, 28]              80

              ReLU-2            [-1, 8, 28, 28]               0

       BatchNorm2d-3            [-1, 8, 28, 28]              16

         Dropout2d-4            [-1, 8, 28, 28]               0

            Conv2d-5            [-1, 8, 28, 28]             584

              ReLU-6            [-1, 8, 28, 28]               0

       BatchNorm2d-7            [-1, 8, 28, 28]              16

         Dropout2d-8            [-1, 8, 28, 28]               0

         MaxPool2d-9            [-1, 8, 14, 14]               0

           Conv2d-10           [-1, 12, 12, 12]             876

             ReLU-11           [-1, 12, 12, 12]               0

      BatchNorm2d-12           [-1, 12, 12, 12]              24

        Dropout2d-13           [-1, 12, 12, 12]               0

           Conv2d-14           [-1, 16, 10, 10]           1,744

             ReLU-15           [-1, 16, 10, 10]               0

      BatchNorm2d-16           [-1, 16, 10, 10]              32

        Dropout2d-17           [-1, 16, 10, 10]               0

        MaxPool2d-18             [-1, 16, 5, 5]               0

           Conv2d-19             [-1, 20, 3, 3]           2,900

             ReLU-20             [-1, 20, 3, 3]               0

      BatchNorm2d-21             [-1, 20, 3, 3]              40

        Dropout2d-22             [-1, 20, 3, 3]               0

        AvgPool2d-23             [-1, 20, 1, 1]               0

           Linear-24                   [-1, 10]             210

================================================================

Total params: 6,522

Trainable params: 6,522

Non-trainable params: 0

----------------------------------------------------------------

Input size (MB): 0.00

Forward/backward pass size (MB): 0.51

Params size (MB): 0.02

Estimated Total Size (MB): 0.53

----------------------------------------------------------------
```

We can monitor our model performance while it's getting trained. The output looks like this - 
```
Epoch 1
Train: Loss=0.0172 Batch_id=937 Accuracy=93.55: 100%|██████████| 938/938 [00:36<00:00, 25.80it/s]
Test set: Average loss: 0.0419, Accuracy: 9859/10000 (98.59%)



Epoch 2
Train: Loss=0.1250 Batch_id=937 Accuracy=97.14: 100%|██████████| 938/938 [00:35<00:00, 26.62it/s]
Test set: Average loss: 0.0336, Accuracy: 9892/10000 (98.92%)



Epoch 3
Train: Loss=0.0043 Batch_id=937 Accuracy=97.57: 100%|██████████| 938/938 [00:36<00:00, 25.54it/s]
Test set: Average loss: 0.0283, Accuracy: 9908/10000 (99.08%)



Epoch 4
Train: Loss=0.1071 Batch_id=937 Accuracy=97.79: 100%|██████████| 938/938 [00:35<00:00, 26.53it/s]
Test set: Average loss: 0.0285, Accuracy: 9913/10000 (99.13%)



Epoch 5
Train: Loss=0.0172 Batch_id=937 Accuracy=98.05: 100%|██████████| 938/938 [00:35<00:00, 26.26it/s]
Test set: Average loss: 0.0255, Accuracy: 9924/10000 (99.24%)



Epoch 6
Train: Loss=0.0459 Batch_id=937 Accuracy=98.06: 100%|██████████| 938/938 [00:36<00:00, 25.72it/s]
Test set: Average loss: 0.0241, Accuracy: 9923/10000 (99.23%)



Epoch 7
Train: Loss=0.0316 Batch_id=937 Accuracy=98.19: 100%|██████████| 938/938 [00:35<00:00, 26.40it/s]
Test set: Average loss: 0.0249, Accuracy: 9928/10000 (99.28%)



Epoch 8
Train: Loss=0.0595 Batch_id=937 Accuracy=98.22: 100%|██████████| 938/938 [00:38<00:00, 24.65it/s]
Test set: Average loss: 0.0244, Accuracy: 9925/10000 (99.25%)



Epoch 00008: reducing learning rate of group 0 to 1.0000e-02.

Epoch 9
Train: Loss=0.0114 Batch_id=937 Accuracy=98.50: 100%|██████████| 938/938 [00:36<00:00, 25.45it/s]
Test set: Average loss: 0.0209, Accuracy: 9938/10000 (99.38%)



Epoch 10
Train: Loss=0.0031 Batch_id=937 Accuracy=98.59: 100%|██████████| 938/938 [00:39<00:00, 23.92it/s]
Test set: Average loss: 0.0205, Accuracy: 9936/10000 (99.36%)



Epoch 11
Train: Loss=0.0290 Batch_id=937 Accuracy=98.64: 100%|██████████| 938/938 [00:36<00:00, 25.81it/s]
Test set: Average loss: 0.0201, Accuracy: 9936/10000 (99.36%)



Epoch 12
Train: Loss=0.0089 Batch_id=937 Accuracy=98.60: 100%|██████████| 938/938 [00:42<00:00, 21.86it/s]
Test set: Average loss: 0.0204, Accuracy: 9941/10000 (99.41%)



Epoch 00012: reducing learning rate of group 0 to 1.0000e-03.

Epoch 13
Train: Loss=0.0217 Batch_id=937 Accuracy=98.67: 100%|██████████| 938/938 [00:42<00:00, 21.82it/s]
Test set: Average loss: 0.0199, Accuracy: 9940/10000 (99.40%)



Epoch 14
Train: Loss=0.0976 Batch_id=937 Accuracy=98.73: 100%|██████████| 938/938 [00:42<00:00, 22.11it/s]
Test set: Average loss: 0.0199, Accuracy: 9937/10000 (99.37%)



Epoch 15
Train: Loss=0.1151 Batch_id=937 Accuracy=98.72: 100%|██████████| 938/938 [00:44<00:00, 20.97it/s]
Test set: Average loss: 0.0197, Accuracy: 9936/10000 (99.36%)

```
## utils.py
The file contains utility & helper functions needed for training & for evaluating our model.

## S7.ipynb
The file is an IPython notebook. The notebook imports helper functions from utils.py and Model class from Model_1.py, Model_2.py & Model_3.py.

## How to setup
### Prerequisits
```
1. python 3.8 or higher
2. pip 22 or higher
```

It's recommended to use virtualenv so that there's no conflict of package versions if there are multiple projects configured on a single system. 
Read more about [virtualenv](https://virtualenv.pypa.io/en/latest/). 

Once virtualenv is activated (or otherwise not opted), install required packages using following command. 

```
pip install requirements.txt
```

## Running IPython Notebook using jupyter
To run the notebook locally -
```
$> cd <to the project folder>
$> jupyter notebook
```
The jupyter server starts with the following output -
```
To access the notebook, open this file in a browser:
        file:///<path to home folder>/Library/Jupyter/runtime/nbserver-71178-open.html
    Or copy and paste one of these URLs:
        http://localhost:8888/?token=64bfa8105e212068866f24d651f51d2b1d4cc6e2627fad41
     or http://127.0.0.1:8888/?token=64bfa8105e212068866f24d651f51d2b1d4cc6e2627fad41
```

Open the above link in your favourite browser, a page similar to below shall be loaded.

![Jupyter server index page](https://github.com/piygr/s5erav1/assets/135162847/40087757-4c99-4b98-8abd-5c4ce95eda38)

- Click on the notebook (.ipynb) link.

A page similar to below shall be loaded. Make sure, it shows *trusted* in top bar. 
If it's not _trusted_, click on *Trust* button and add to the trusted files.

![Jupyter notebook page](https://github.com/piygr/s5erav1/assets/135162847/7858da8f-e07e-47cd-9aa9-19c8c569def1)
Now, the notebook can be operated from the action panel.

Happy Modeling :-) 
 
