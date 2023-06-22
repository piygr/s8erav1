# Session 6 Assignment - Part 1
![Neural Network](https://github.com/piygr/s6erav1/assets/135162847/7ac22310-965b-4cba-930e-65ddee3e6c45)
**Showcasing backpropogation through excel for the above mentioned neural network**

## How bakcpropogation works?
The goal of neural netowrk training is to reduce the network error (mean squared error) of estimnated output with the labeled output.
So, with every delta change in error, the mechanism of updating the network parameters value is called backpropogation.
With the given neural network, we will understand how the backpropogation looks in action.

1. In the first step, every neuron as a function of previous layer's connected neurons & corresponding weights is defined. Refer 1.) in the below screenshot.

<img width="1337" alt="s6_part1_dEdW" src="https://github.com/piygr/s6erav1/assets/135162847/101f7fae-431a-4bc7-be26-2e6fecc9f587">

2. In the second step, the partial derivatives of error w.r.t. parameters (weights), starting from the output layer moving towards the input layer (hence backpropogation) are derived. Refer to 2.) - 6.). 

3. Once all the partial derivatives are derived in terms of weights, input values, target values & previous layer's neuron values, the excel sheet is populated as show below. 
<img width="1222" alt="s6_part1_layer1" src="https://github.com/piygr/s6erav1/assets/135162847/151190c1-57f9-4209-bc18-e160fc4bf94e">
<img width="1125" alt="s6_part1_layer2" src="https://github.com/piygr/s6erav1/assets/135162847/5cac1f43-4355-45c1-a042-bbf6d5af992f">
<img width="914" alt="s6_part1_error_gradients" src="https://github.com/piygr/s6erav1/assets/135162847/afa864a0-9ec9-4b40-b7e9-dff862a40436">

4. With every iteration, all the weights are updated, all neurons are computed and with the new error, the process is continued.

5. Note that, to update weights in every iteration, the following equation is used. Here, learning rate plays an important role.
 
![image](https://github.com/piygr/s6erav1/assets/135162847/99f89b6c-e2f9-421b-826d-210ca6c1c61b)  

## Effects of learning rate

<table>
        <tr>
                <td>
                        <span>
        <b>Learning Rate - 0.1</b><br/>
        <img width="300" alt="s6_part1_lr0 1" src="https://github.com/piygr/s6erav1/assets/135162847/39d84d65-3271-46c9-b048-b6c89a4473db">
</span>
                </td>
                <td>
                        <span><b>Learning Rate - 0.2</b><br/><img width="300" alt="s6_part1_lr0 2" src="https://github.com/piygr/s6erav1/assets/135162847/fae4258a-73d6-4cff-8aac-fdd53f17ab41"></span>
                <td>
        </tr>
        <tr>
                <td>
                        <span>
        <b>Learning Rate - 0.5</b><br/>
        <img width="300" alt="s6_part1_lr0 5" src="https://github.com/piygr/s6erav1/assets/135162847/9c0b9cc7-78ac-48db-844f-dcf30f98e15b">
</span> 
                </td>
                <td>
                    <span>
        <b>Learning Rate - 0.8</b><br/>
        <img width="300" alt="s6_part1_lr0 8" src="https://github.com/piygr/s6erav1/assets/135162847/94fc1378-82f9-47d7-be53-d087204c7b3f">
</span>    
                <td>
        </tr>
        <tr>
                <td>
                    <span>
        <b>Learning Rate - 1.0</b><br/>
        <img width="300" alt="s6_part1_lr1 0" src="https://github.com/piygr/s6erav1/assets/135162847/f88d1e41-e976-4265-b2bc-1277438974d6">
</span>    
                </td>
                <td>
                     <span>
        <b>Learning Rate - 2.0</b><br/>
        <img width="300" alt="s6_part1_lr2 0" src="https://github.com/piygr/s6erav1/assets/135162847/7099b159-1a1d-4632-a010-c78c157529cc">
</span>
                <td>
        </tr>
</table>

Since, learning rate is nothing but the weightage given to the error gradient. Higher the learning rate, faster the error converges to zero.
But after a point if the learning rate is increased, the error doesn't converge. The reason is, the weights never reach to their local minima.

-----

# Session 6 Assignment - Part 2
Model to detect handwritten digits, trained on MNIST dataset of 60,000 images.

**Goal is to create a model with**
- 99.4% validation accuracy
- Less than 20k Parameters
- Less than 20 Epochs
- Have used BN, Dropout,
- (Optional): a Fully connected layer, have used GAP.

## model.py
The file contains model class *OptimizedNet* as subclass of _torch.nn.Module_. The _OptimizedNet_ model has 3 convolution blocks, followed by GAP layer.

Below is the model summary -
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
           Conv2d-10           [-1, 16, 14, 14]           1,168
             ReLU-11           [-1, 16, 14, 14]               0
      BatchNorm2d-12           [-1, 16, 14, 14]              32
        Dropout2d-13           [-1, 16, 14, 14]               0
           Conv2d-14           [-1, 16, 14, 14]           2,320
             ReLU-15           [-1, 16, 14, 14]               0
      BatchNorm2d-16           [-1, 16, 14, 14]              32
        Dropout2d-17           [-1, 16, 14, 14]               0
           Conv2d-18           [-1, 32, 12, 12]           4,640
             ReLU-19           [-1, 32, 12, 12]               0
      BatchNorm2d-20           [-1, 32, 12, 12]              64
        Dropout2d-21           [-1, 32, 12, 12]               0
        MaxPool2d-22             [-1, 32, 6, 6]               0
           Conv2d-23             [-1, 32, 4, 4]           9,248
             ReLU-24             [-1, 32, 4, 4]               0
      BatchNorm2d-25             [-1, 32, 4, 4]              64
        Dropout2d-26             [-1, 32, 4, 4]               0
           Conv2d-27             [-1, 10, 4, 4]             330
================================================================
Total params: 18,594
Trainable params: 18,594
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.00
Forward/backward pass size (MB): 0.75
Params size (MB): 0.07
Estimated Total Size (MB): 0.83
----------------------------------------------------------------
```

## utils.py
The file contains utility & helper functions needed for training & for evaluating our model.

## S6.ipynb
The file is an IPython notebook. The notebook imports helper functions & _OptimizedNet_ model class from utils.py and model.py respectively.
In the notebook, we are creating train & test datasets with various transformations on the base MNIST dataset.

We can monitor our model performance while it's getting trained. The output looks like this - 
```
Adjusting learning rate of group 0 to 1.0000e-02.
Epoch 1
Train: Loss=0.1356 Batch_id=937 Accuracy=92.10: 100%|██████████| 938/938 [00:47<00:00, 19.96it/s]
Test set: Average loss: 0.0456, Accuracy: 9868/10000 (98.68%)

Adjusting learning rate of group 0 to 1.0000e-02.
Epoch 2
Train: Loss=0.0651 Batch_id=937 Accuracy=97.64: 100%|██████████| 938/938 [00:49<00:00, 19.09it/s]
Test set: Average loss: 0.0368, Accuracy: 9881/10000 (98.81%)

Adjusting learning rate of group 0 to 1.0000e-02.
Epoch 3
Train: Loss=0.0078 Batch_id=937 Accuracy=98.07: 100%|██████████| 938/938 [00:40<00:00, 23.30it/s]
Test set: Average loss: 0.0215, Accuracy: 9930/10000 (99.30%)

Adjusting learning rate of group 0 to 1.0000e-02.
Epoch 4
Train: Loss=0.0287 Batch_id=937 Accuracy=98.25: 100%|██████████| 938/938 [00:44<00:00, 21.14it/s]
Test set: Average loss: 0.0206, Accuracy: 9934/10000 (99.34%)

Adjusting learning rate of group 0 to 1.0000e-02.
Epoch 5
Train: Loss=0.0379 Batch_id=937 Accuracy=98.53: 100%|██████████| 938/938 [00:44<00:00, 21.15it/s]
Test set: Average loss: 0.0208, Accuracy: 9930/10000 (99.30%)

Adjusting learning rate of group 0 to 1.0000e-02.
Epoch 6
Train: Loss=0.0112 Batch_id=937 Accuracy=98.54: 100%|██████████| 938/938 [00:45<00:00, 20.62it/s]
Test set: Average loss: 0.0185, Accuracy: 9937/10000 (99.37%)

Adjusting learning rate of group 0 to 1.0000e-02.
Epoch 7
Train: Loss=0.0785 Batch_id=937 Accuracy=98.61: 100%|██████████| 938/938 [00:45<00:00, 20.56it/s]
Test set: Average loss: 0.0190, Accuracy: 9942/10000 (99.42%)

Adjusting learning rate of group 0 to 1.0000e-03.
Epoch 8
Train: Loss=0.0279 Batch_id=937 Accuracy=98.87: 100%|██████████| 938/938 [00:43<00:00, 21.38it/s]
Test set: Average loss: 0.0150, Accuracy: 9948/10000 (99.48%)

Adjusting learning rate of group 0 to 1.0000e-03.
Epoch 9
Train: Loss=0.1910 Batch_id=937 Accuracy=98.95: 100%|██████████| 938/938 [00:43<00:00, 21.52it/s]
Test set: Average loss: 0.0149, Accuracy: 9954/10000 (99.54%)

Adjusting learning rate of group 0 to 1.0000e-03.
Epoch 10
Train: Loss=0.0464 Batch_id=937 Accuracy=98.92: 100%|██████████| 938/938 [00:43<00:00, 21.32it/s]
Test set: Average loss: 0.0145, Accuracy: 9950/10000 (99.50%)

Adjusting learning rate of group 0 to 1.0000e-03.
Epoch 11
Train: Loss=0.0013 Batch_id=937 Accuracy=99.00: 100%|██████████| 938/938 [00:47<00:00, 19.81it/s]
Test set: Average loss: 0.0143, Accuracy: 9952/10000 (99.52%)

Adjusting learning rate of group 0 to 1.0000e-03.
Epoch 12
Train: Loss=0.1712 Batch_id=937 Accuracy=99.06: 100%|██████████| 938/938 [00:44<00:00, 21.30it/s]
Test set: Average loss: 0.0138, Accuracy: 9953/10000 (99.53%)

Adjusting learning rate of group 0 to 1.0000e-03.
Epoch 13
Train: Loss=0.0084 Batch_id=937 Accuracy=99.06: 100%|██████████| 938/938 [00:44<00:00, 21.24it/s]
Test set: Average loss: 0.0135, Accuracy: 9955/10000 (99.55%)

Adjusting learning rate of group 0 to 1.0000e-03.
Epoch 14
Train: Loss=0.0052 Batch_id=937 Accuracy=99.13: 100%|██████████| 938/938 [00:44<00:00, 21.00it/s]
Test set: Average loss: 0.0132, Accuracy: 9956/10000 (99.56%)

Adjusting learning rate of group 0 to 1.0000e-04.
Epoch 15
Train: Loss=0.0142 Batch_id=937 Accuracy=99.06: 100%|██████████| 938/938 [00:43<00:00, 21.55it/s]
Test set: Average loss: 0.0131, Accuracy: 9955/10000 (99.55%)

Adjusting learning rate of group 0 to 1.0000e-04.
Epoch 16
Train: Loss=0.0187 Batch_id=937 Accuracy=98.97: 100%|██████████| 938/938 [00:44<00:00, 20.92it/s]
Test set: Average loss: 0.0132, Accuracy: 9957/10000 (99.57%)

Adjusting learning rate of group 0 to 1.0000e-04.
Epoch 17
Train: Loss=0.0015 Batch_id=937 Accuracy=99.11: 100%|██████████| 938/938 [00:45<00:00, 20.62it/s]
Test set: Average loss: 0.0129, Accuracy: 9960/10000 (99.60%)

Adjusting learning rate of group 0 to 1.0000e-04.
Epoch 18
Train: Loss=0.0042 Batch_id=937 Accuracy=99.08: 100%|██████████| 938/938 [00:45<00:00, 20.57it/s]
Test set: Average loss: 0.0129, Accuracy: 9957/10000 (99.57%)

Adjusting learning rate of group 0 to 1.0000e-04.
Epoch 19
Train: Loss=0.2548 Batch_id=937 Accuracy=99.04: 100%|██████████| 938/938 [00:45<00:00, 20.84it/s]
Test set: Average loss: 0.0132, Accuracy: 9954/10000 (99.54%)

Adjusting learning rate of group 0 to 1.0000e-04.
Epoch 20
Train: Loss=0.0566 Batch_id=937 Accuracy=99.08: 100%|██████████| 938/938 [00:43<00:00, 21.37it/s]
Test set: Average loss: 0.0131, Accuracy: 9958/10000 (99.58%)

Adjusting learning rate of group 0 to 1.0000e-04.
```  

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

## Running IPython Notebook (S6.ipynb) using jupyter
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
 
