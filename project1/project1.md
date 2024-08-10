# About the dataset 
The Modified National Institute of Standards and Technology (MNIST) dataset is a collection of 70,000 grayscale handwritten digits (0-9), with each image being 28×28 pixels. It has a training set of 60,000 examples and a test set of 10,000 examples. The digits have been size-normalized and centered in a fixed-size image.

# TensorFlow implementation
- Keras is a high-level API used for building and training neural networks.
- Sequential is a class that allows us to create a sequential model, where layers are stacked on top of each other.
- Dense is a class that represents a fully connected layer in a neural network.
- Flatten is a class that flattens the input into a 1-dimensional array.
- mnist is a module that provides the MNIST dataset.
- to_categorical is a function that converts the labels into one-hot encoded vectors.

## Loading and preprocessing
- The MNIST dataset is loaded through `mnist.load_data()` and split into training and testing sets.
- The pixel values of the images are normalized by dividing them by 255.0, which scales them to the range of 0 to 1 for faster training.
- The labels are converted to one-hot encoded vectors using `to_categorical`.

## Defining the model
- Sequential is used to create a sequential model.
- A Flatten layer is added as the first layer to flatten the input images from a 2D shape (28x28) to a 1D shape (784).
- Dense layers are added to the model. The first dense layer has 128 units and uses the ReLU activation function. The second dense layer has 10 units (corresponding to the 10 possible digit classes) and uses the softmax activation function.
- The model is configured with the Adam optimizer, categorical cross-entropy loss, and accuracy as the evaluation metric.
- `with tf.device('/GPU:0'):` is used to specify that the model should be trained on the GPU if available. This can significantly speed up training for large models and datasets.
- The model is trained using `model.fit()`.
- The model is evaluated using `model.evaluate()`.

# PyTorch implementation
- The training and test datasets are loaded and transformed. `transforms.Compose([...])` allows you to chain multiple image transformations together. The transformations are applied in the order they are listed. `transforms.ToTensor()` converts a PIL Image or a NumPy ndarray into a PyTorch tensor. The image data is scaled from the range [0, 255] to [0.0, 1.0]. `transforms.Normalize((0.5,), (0.5,))` normalizes the tensor image with mean and standard deviation. The values (0.5,) for mean and (0.5,) for standard deviation scale the data to have a mean of 0.5 and a standard deviation of 0.5, effectively transforming the range from [0.0, 1.0] to [-1.0, 1.0] (image = (image - mean) / std).
- Data loaders are created for training and testing.

## Defining the neural network model
- The SimpleNN class is a subclass of nn.Module, which is the base class for all neural network modules in PyTorch.
- In the constructor (__init__), the layers of the neural network are defined. Here, we have a flattening layer (nn.Flatten()), followed by two fully connected layers (nn.Linear()).
- The forward method defines the forward pass of the neural network. It takes an input x and applies the layers in sequence to produce the output x.
- An instance of the SimpleNN class is created and assigned to the model variable.
- The `nn.CrossEntropyLoss()` function is used as the loss function. The `optim.Adam()` function is used as the optimizer. It implements the Adam optimization algorithm and is used to update the model's parameters during training.
- The code checks if a GPU is available using `torch.cuda.is_available()`. If a GPU is available, the model is moved to the GPU using the `to()` method.
- The model is trained for a specified number of epochs (in this case, 5).
- Inside the training loop, the model is set to training mode using `model.train()`.
- The training data is iterated over in batches using the `train_loader`.
- For each batch, the input images and labels are moved to the GPU if available.
- The forward pass is performed by passing the input images through the model to obtain the predicted outputs.
- The loss is calculated by comparing the predicted outputs with the ground truth labels using the specified loss function.
- The optimizer's gradients are reset to zero using `optimizer.zero_grad()`.
- The gradients are computed using backpropagation with `loss.backward()`.
- The optimizer updates the model's parameters using `optimizer.step()`.
- The loss for the current epoch is printed.

# References
- [GeeksForGeeks](https://www.geeksforgeeks.org/mnist-dataset/#what-is-mnist-dataset)
- [Yann LeCun](https://yann.lecun.com/exdb/mnist/)
