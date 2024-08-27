2024.8.16 ： This is a deep learning framework based on numpy and math. More information may be added later.<br>
2024.8.19： Calib now can support more Tensor operations such as concatenate, stack, split, flip, dilation, and convolution(both forward and backward<br>
            the the correctness of forward pass and backward pass was tested by comparing the coresponding result of pytorch.<br>
            The simple_exercise is a demo about how to use neural network to approximate functions.<br>
            The simple_exercise_cnn is a example of convolutional neural networks, if you want to ues it, please download the cifar-10 dataset(python version), and create a new folder in calib named "data"，put the dataset in the "data" folder and unzip it.
            Another way is to put this unzipped dataset into any folder(as you like) and change the file direction in simple_exercise_cnn.py to read the coresponding file in that path. 

2024.8.27：RNN, LSTM, Transformer are enabled(added in nn.py).
