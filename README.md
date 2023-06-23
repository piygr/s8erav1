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
                <td>Group Normalization (No of groups G := 4) </td>
                <td>73.54%</td>
                <td>73.00%</td>
                <td>48,178</td>
        </tr>
        <tr>
                <td>Layer Normalization ( G := 1) </td>
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
2. mean and sigma values are computed for every group. Therefore, number of untrainable parameters = 2 * G. But trainable parameters (Scale & Shift parametrs) are stored w.r.t. every channel. Every channel belonging to a group *g* will share same trainable parameters but will have different trainable (Scale & Shift) parameters.

Below is the model summary -
```
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1           [-1, 16, 32, 32]             448
         GroupNorm-2           [-1, 16, 32, 32]              32
         Dropout2d-3           [-1, 16, 32, 32]               0
            Conv2d-4           [-1, 16, 32, 32]           2,320
         GroupNorm-5           [-1, 16, 32, 32]              32
         Dropout2d-6           [-1, 16, 32, 32]               0
            Conv2d-7            [-1, 8, 32, 32]             136
         MaxPool2d-8            [-1, 8, 16, 16]               0
            Conv2d-9           [-1, 32, 16, 16]           2,336
        GroupNorm-10           [-1, 32, 16, 16]              64
        Dropout2d-11           [-1, 32, 16, 16]               0
           Conv2d-12           [-1, 32, 16, 16]           9,248
        GroupNorm-13           [-1, 32, 16, 16]              64
        Dropout2d-14           [-1, 32, 16, 16]               0
           Conv2d-15           [-1, 32, 16, 16]           9,248
        GroupNorm-16           [-1, 32, 16, 16]              64
        Dropout2d-17           [-1, 32, 16, 16]               0
           Conv2d-18           [-1, 16, 16, 16]             528
        MaxPool2d-19             [-1, 16, 8, 8]               0
           Conv2d-20             [-1, 32, 8, 8]           4,640
        GroupNorm-21             [-1, 32, 8, 8]              64
        Dropout2d-22             [-1, 32, 8, 8]               0
           Conv2d-23             [-1, 32, 8, 8]           9,248
        GroupNorm-24             [-1, 32, 8, 8]              64
        Dropout2d-25             [-1, 32, 8, 8]               0
           Conv2d-26             [-1, 32, 8, 8]           9,248
        GroupNorm-27             [-1, 32, 8, 8]              64
        Dropout2d-28             [-1, 32, 8, 8]               0
        AvgPool2d-29             [-1, 32, 1, 1]               0
           Conv2d-30             [-1, 10, 1, 1]             330
================================================================
Total params: 48,178
Trainable params: 48,178
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.01
Forward/backward pass size (MB): 1.57
Params size (MB): 0.18
Estimated Total Size (MB): 1.77
----------------------------------------------------------------
```

We can monitor our model performance while it's getting trained. The output looks like this - 
```
Adjusting learning rate of group 0 to 1.0000e-02.
Epoch 1
Train: Loss=1.6918 Batch_id=781 Accuracy=31.06: 100%|██████████| 782/782 [01:49<00:00,  7.14it/s]
Test set: Average loss: 1.5039, Accuracy: 4398/10000 (43.98%)

Adjusting learning rate of group 0 to 1.0000e-02.
Epoch 2
Train: Loss=0.9059 Batch_id=781 Accuracy=47.94: 100%|██████████| 782/782 [01:45<00:00,  7.45it/s]
Test set: Average loss: 1.2714, Accuracy: 5324/10000 (53.24%)

Adjusting learning rate of group 0 to 1.0000e-02.
Epoch 3
Train: Loss=0.9463 Batch_id=781 Accuracy=55.08: 100%|██████████| 782/782 [01:46<00:00,  7.36it/s]
Test set: Average loss: 1.1538, Accuracy: 5804/10000 (58.04%)

Adjusting learning rate of group 0 to 1.0000e-02.
Epoch 4
Train: Loss=0.7531 Batch_id=781 Accuracy=59.48: 100%|██████████| 782/782 [01:50<00:00,  7.10it/s]
Test set: Average loss: 1.0502, Accuracy: 6167/10000 (61.67%)

Adjusting learning rate of group 0 to 1.0000e-02.
Epoch 5
Train: Loss=0.9614 Batch_id=781 Accuracy=62.61: 100%|██████████| 782/782 [01:55<00:00,  6.80it/s]
Test set: Average loss: 1.0070, Accuracy: 6415/10000 (64.15%)

Adjusting learning rate of group 0 to 1.0000e-02.
Epoch 6
Train: Loss=0.5281 Batch_id=781 Accuracy=65.24: 100%|██████████| 782/782 [01:45<00:00,  7.40it/s]
Test set: Average loss: 0.9796, Accuracy: 6537/10000 (65.37%)

Adjusting learning rate of group 0 to 1.0000e-02.
Epoch 7
Train: Loss=1.1028 Batch_id=781 Accuracy=67.28: 100%|██████████| 782/782 [01:50<00:00,  7.05it/s]
Test set: Average loss: 0.8961, Accuracy: 6845/10000 (68.45%)

Adjusting learning rate of group 0 to 1.0000e-03.
Epoch 8
Train: Loss=0.5350 Batch_id=781 Accuracy=71.80: 100%|██████████| 782/782 [01:52<00:00,  6.96it/s]
Test set: Average loss: 0.7926, Accuracy: 7151/10000 (71.51%)

Adjusting learning rate of group 0 to 1.0000e-03.
Epoch 9
Train: Loss=0.5492 Batch_id=781 Accuracy=72.47: 100%|██████████| 782/782 [01:50<00:00,  7.08it/s]
Test set: Average loss: 0.7879, Accuracy: 7163/10000 (71.63%)

Adjusting learning rate of group 0 to 1.0000e-03.
Epoch 10
Train: Loss=0.7824 Batch_id=781 Accuracy=72.86: 100%|██████████| 782/782 [01:50<00:00,  7.09it/s]
Test set: Average loss: 0.7889, Accuracy: 7186/10000 (71.86%)

Adjusting learning rate of group 0 to 1.0000e-03.
Epoch 11
Train: Loss=0.9571 Batch_id=781 Accuracy=73.39: 100%|██████████| 782/782 [01:53<00:00,  6.87it/s]
Test set: Average loss: 0.7754, Accuracy: 7229/10000 (72.29%)

Adjusting learning rate of group 0 to 1.0000e-03.
Epoch 12
Train: Loss=0.8656 Batch_id=781 Accuracy=73.54: 100%|██████████| 782/782 [01:53<00:00,  6.86it/s]
Test set: Average loss: 0.7633, Accuracy: 7300/10000 (73.00%)

```
## Layer Normalization

### Observation
1. Layer normalization is a special case of Group normalization. With G = 1, GroupNorm is nothing but Layer normalization.
2. mean and sigma values are computed for the group means across all the channels. Therefore, number of untrainable parameters = 2. But trainable parameters (Scale & Shift parametrs) are stored w.r.t. every channel.
3. 

Below is the model summary -
```
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1           [-1, 16, 32, 32]             448
         GroupNorm-2           [-1, 16, 32, 32]              32
         Dropout2d-3           [-1, 16, 32, 32]               0
            Conv2d-4           [-1, 16, 32, 32]           2,320
         GroupNorm-5           [-1, 16, 32, 32]              32
         Dropout2d-6           [-1, 16, 32, 32]               0
            Conv2d-7            [-1, 8, 32, 32]             136
         MaxPool2d-8            [-1, 8, 16, 16]               0
            Conv2d-9           [-1, 32, 16, 16]           2,336
        GroupNorm-10           [-1, 32, 16, 16]              64
        Dropout2d-11           [-1, 32, 16, 16]               0
           Conv2d-12           [-1, 32, 16, 16]           9,248
        GroupNorm-13           [-1, 32, 16, 16]              64
        Dropout2d-14           [-1, 32, 16, 16]               0
           Conv2d-15           [-1, 32, 16, 16]           9,248
        GroupNorm-16           [-1, 32, 16, 16]              64
        Dropout2d-17           [-1, 32, 16, 16]               0
           Conv2d-18           [-1, 16, 16, 16]             528
        MaxPool2d-19             [-1, 16, 8, 8]               0
           Conv2d-20             [-1, 32, 8, 8]           4,640
        GroupNorm-21             [-1, 32, 8, 8]              64
        Dropout2d-22             [-1, 32, 8, 8]               0
           Conv2d-23             [-1, 32, 8, 8]           9,248
        GroupNorm-24             [-1, 32, 8, 8]              64
        Dropout2d-25             [-1, 32, 8, 8]               0
           Conv2d-26             [-1, 32, 8, 8]           9,248
        GroupNorm-27             [-1, 32, 8, 8]              64
        Dropout2d-28             [-1, 32, 8, 8]               0
        AvgPool2d-29             [-1, 32, 1, 1]               0
           Conv2d-30             [-1, 10, 1, 1]             330
================================================================
Total params: 48,178
Trainable params: 48,178
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.01
Forward/backward pass size (MB): 1.57
Params size (MB): 0.18
Estimated Total Size (MB): 1.77
----------------------------------------------------------------
```

We can monitor our model performance while it's getting trained. The output looks like this - 
```
Adjusting learning rate of group 0 to 1.0000e-02.

Epoch 1
Train: Loss=2.0968 Batch_id=781 Accuracy=30.14: 100%|██████████| 782/782 [01:48<00:00,  7.24it/s]
Test set: Average loss: 1.5972, Accuracy: 3725/10000 (37.25%)

Adjusting learning rate of group 0 to 1.0000e-02.
Epoch 2
Train: Loss=1.2773 Batch_id=781 Accuracy=44.44: 100%|██████████| 782/782 [01:45<00:00,  7.41it/s]
Test set: Average loss: 1.3278, Accuracy: 5176/10000 (51.76%)

Adjusting learning rate of group 0 to 1.0000e-02.
Epoch 3
Train: Loss=1.0092 Batch_id=781 Accuracy=52.26: 100%|██████████| 782/782 [01:52<00:00,  6.95it/s]
Test set: Average loss: 1.1520, Accuracy: 5808/10000 (58.08%)

Adjusting learning rate of group 0 to 1.0000e-02.
Epoch 4
Train: Loss=0.6062 Batch_id=781 Accuracy=57.36: 100%|██████████| 782/782 [01:47<00:00,  7.26it/s]
Test set: Average loss: 1.0766, Accuracy: 6083/10000 (60.83%)

Adjusting learning rate of group 0 to 1.0000e-02.
Epoch 5
Train: Loss=1.3678 Batch_id=781 Accuracy=61.50: 100%|██████████| 782/782 [01:43<00:00,  7.58it/s]
Test set: Average loss: 1.0043, Accuracy: 6316/10000 (63.16%)

Adjusting learning rate of group 0 to 1.0000e-02.
Epoch 6
Train: Loss=1.0654 Batch_id=781 Accuracy=63.98: 100%|██████████| 782/782 [01:47<00:00,  7.30it/s]
Test set: Average loss: 0.9630, Accuracy: 6548/10000 (65.48%)

Adjusting learning rate of group 0 to 1.0000e-02.
Epoch 7
Train: Loss=0.7331 Batch_id=781 Accuracy=66.08: 100%|██████████| 782/782 [01:42<00:00,  7.60it/s]
Test set: Average loss: 0.8860, Accuracy: 6839/10000 (68.39%)

Adjusting learning rate of group 0 to 1.0000e-03.
Epoch 8
Train: Loss=0.4923 Batch_id=781 Accuracy=71.19: 100%|██████████| 782/782 [01:47<00:00,  7.30it/s]
Test set: Average loss: 0.8234, Accuracy: 7042/10000 (70.42%)

Adjusting learning rate of group 0 to 1.0000e-03.
Epoch 9
Train: Loss=0.7708 Batch_id=781 Accuracy=71.98: 100%|██████████| 782/782 [01:40<00:00,  7.77it/s]
Test set: Average loss: 0.8105, Accuracy: 7137/10000 (71.37%)

Adjusting learning rate of group 0 to 1.0000e-03.
Epoch 10
Train: Loss=0.7846 Batch_id=781 Accuracy=72.37: 100%|██████████| 782/782 [01:40<00:00,  7.80it/s]
Test set: Average loss: 0.8103, Accuracy: 7147/10000 (71.47%)

Adjusting learning rate of group 0 to 1.0000e-03.
Epoch 11
Train: Loss=0.7809 Batch_id=781 Accuracy=72.56: 100%|██████████| 782/782 [01:39<00:00,  7.85it/s]
Test set: Average loss: 0.7951, Accuracy: 7160/10000 (71.60%)

Adjusting learning rate of group 0 to 1.0000e-03.
Epoch 12
Train: Loss=0.4910 Batch_id=781 Accuracy=72.95: 100%|██████████| 782/782 [01:44<00:00,  7.50it/s]
Test set: Average loss: 0.7954, Accuracy: 7148/10000 (71.48%)

Adjusting learning rate of group 0 to 1.0000e-03.
Epoch 13
Train: Loss=0.4693 Batch_id=781 Accuracy=73.40: 100%|██████████| 782/782 [01:49<00:00,  7.11it/s]
Test set: Average loss: 0.7941, Accuracy: 7164/10000 (71.64%)

Adjusting learning rate of group 0 to 1.0000e-03.
Epoch 14
Train: Loss=0.7188 Batch_id=781 Accuracy=73.62: 100%|██████████| 782/782 [01:48<00:00,  7.20it/s]
Test set: Average loss: 0.7792, Accuracy: 7213/10000 (72.13%)

Adjusting learning rate of group 0 to 1.0000e-04.
Epoch 15
Train: Loss=0.6309 Batch_id=781 Accuracy=74.18: 100%|██████████| 782/782 [01:42<00:00,  7.63it/s]
Test set: Average loss: 0.7696, Accuracy: 7258/10000 (72.58%)

Adjusting learning rate of group 0 to 1.0000e-04.
Epoch 16
Train: Loss=0.9905 Batch_id=781 Accuracy=74.17: 100%|██████████| 782/782 [01:43<00:00,  7.59it/s]
Test set: Average loss: 0.7693, Accuracy: 7255/10000 (72.55%)

Adjusting learning rate of group 0 to 1.0000e-04.
Epoch 17
Train: Loss=0.6600 Batch_id=781 Accuracy=74.62: 100%|██████████| 782/782 [01:40<00:00,  7.75it/s]
Test set: Average loss: 0.7682, Accuracy: 7264/10000 (72.64%)

Adjusting learning rate of group 0 to 1.0000e-04.
Epoch 18
Train: Loss=0.7382 Batch_id=781 Accuracy=74.53: 100%|██████████| 782/782 [01:40<00:00,  7.82it/s]
Test set: Average loss: 0.7676, Accuracy: 7260/10000 (72.60%)

Adjusting learning rate of group 0 to 1.0000e-04.
Epoch 19
Train: Loss=0.8322 Batch_id=781 Accuracy=74.48: 100%|██████████| 782/782 [01:40<00:00,  7.81it/s]
Test set: Average loss: 0.7674, Accuracy: 7272/10000 (72.72%)

Adjusting learning rate of group 0 to 1.0000e-04.
Epoch 20
Train: Loss=0.3287 Batch_id=781 Accuracy=74.46: 100%|██████████| 782/782 [01:40<00:00,  7.79it/s]
Test set: Average loss: 0.7667, Accuracy: 7272/10000 (72.72%)

```


## utils.py
The file contains utility & helper functions needed for training & for evaluating our model.

## S8.ipynb
The file is an IPython notebook. The notebook imports helper functions from utils.py.

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
 
