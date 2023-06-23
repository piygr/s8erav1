# Session 8 Assignment
Model built on CIFAR10

**Goal is to create a model with**
1. this network: C1 C2 c3 P1 C3 C4 C5 c6 P2 C7 C8 C9 GAP C10
2. Keep the parameter count less than 50000
3. Max Epochs is 20
4. 3 versions of the above code (in each case achieve above 70% accuracy):
 - Network with Group Normalization
 - Network with Layer Normalization
 - Network with Batch Normalization

------
<table>
        <tr>
                <th></th>
                <th>Training Accuracy</th>
                <th>Test Accuracy</th>
                <th>Total Parameters</th>
        </tr>
        <tr>
                <td>Group Normalization (No of groups - 4) </td>
                <td>73.54%</td>
                <td>73.00%</td>
                <td>48,178</td>
        </tr>
        <tr>
                <td>Layer Normalization</td>
                <td>74.46%</td>
                <td>72.72%</td>
                <td>48,178</td>
        </tr>
        <tr>
                <td>Batch Normalization (Batch size - 64) </td>
                <td>73.58%</td>
                <td>74.89%</td>
                <td>48,178</td>
        </tr>
</table>

## Group Normalization

### Observation
1. Number of groups has to be a factor of number of channels. So, if G is number of groups and C is number of input channels, then every group contains C / G number of channels.
2. Though mean and sigma is calculated for every group. Therefore, number of untrainable parameters = 2 * G. But trainable parameters (Scale & Shift parametrs) are computed w.r.t. every channel.

Below is the model () summary -
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
 
