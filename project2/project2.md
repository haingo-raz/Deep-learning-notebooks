# General architecture of CNN models
![FashionMNIST](https://editor.analyticsvidhya.com/uploads/67201cnn.jpeg)
[Source](https://editor.analyticsvidhya.com/uploads/67201cnn.jpeg)

## CNN Architecture
- The convolutional layer applies a set of filters to the input to produce a feature map.
- The activation layer introduces non-linearity into the model using activation functions.
- The pooling layer reduces the spatial dimensions of the feature map while retaining the most important information.
- The fully connected layer combines features learned by the convolutional and pooling layers to classify the input.

# About
FashionMNIST is a popular dataset in computer vision, consisting of 60,000 training images and 10,000 test images. Each image is a grayscale 28x28 pixel image belonging to one of the 10 different classes, representing various fashion items such as T-shirts, dresses, shoes, etc. The goal is to train a CNN model to classify these images into their respective categories.

# Preprocessing
- `transforms.ToTensor()` converts the images in the dataset from PIL Image format to PyTorch Tensor format.
- `transforms.Normalize((0.5,), (0.5,))` normalizes the pixel values of the images to the range [-1,1].
- Download, load, and apply transformations to the training and test sets.

# Architecture of the model
- The convolutional layer conv1 performs convolutional operations on the input image. It uses a kernel size of 3x3 pixels and applies 16 filters. The output of this layer is a feature map with 16 channels.
- The max pooling layer reduces the spatial dimensions of the input while retaining the most important features. In this case, a 2x2 pixel kernel is used with a stride of 2, meaning the pooling window slides by 2 pixels at a time. This reduces the spatial dimensions of the feature map by half.
- The fully-connected layers are fc1 and fc2. The output of the max pooling layer is flattened into a 1-dimensional vector. This flattened vector is then passed through two fully connected layers. The first fully connected layer (self.fc1) takes the flattened vector as input and produces an output of size 128. The second fully connected layer (self.fc2) takes the output of the previous layer and produces the final output with a size of 10. The output size of 10 corresponds to the number of classes in the classification problem.
- The forward method of the Classifier class defines the forward pass of the model. It takes an input tensor x and applies the layers in a sequential manner. The input tensor is passed through the convolutional layer, followed by the max pooling layer. Then, the tensor is reshaped using the view method to match the expected input size of the fully connected layer. The reshaped tensor is passed through the fully connected layers with ReLU activation functions. The final output is returned.

# Training
- The model is instantiated.
- It is configured to use cross entropy as the loss function and SGD with momentum as the optimizer.
- The model is trained for 2 epochs.
- `optimizer.zero_grad()` clears the gradients of all optimized parameters to prevent accumulation from previous iterations.
- `loss.backward()` computes the gradient of the loss with respect to the model's parameters.
- `optimizer.step()` updates the model's parameters based on the computed gradients.

# Evaluation
The model is being evaluated on the test data. The variables correct and total are initialized to keep track of the number of correctly predicted labels and the total number of labels, respectively. The code then iterates over the testloader, which contains batches of test data. For each batch, the model predicts the labels for the images using the model function. The predicted labels are compared with the true labels, and the number of correct predictions is incremented. The total number of labels is also updated. After iterating through all the test data, the accuracy of the model is calculated by dividing the number of correct predictions by the total number of labels and multiplying by 100.

- `torch.no_grad()` is a context manager in PyTorch that disables gradient calculation. This is useful during inference when you are sure that you will not call Tensor.backward(). It reduces memory consumption for computations that would otherwise have requires_grad=True.
- `_, predicted = torch.max(outputs.data, 1)` is used to get the class with the highest prediction probability.
- `total += labels.size(0)` is used to keep track of the total number of samples that have been processed. `labels.size(0)` in the current batch, and it’s added to the total.
- `predicted_labels.extend(predicted.tolist())` and `true_labels.extend(labels.tolist())` are used to store all the predicted labels and true labels for the test dataset.

# Inference
- `img = img / 2 + 0.5` is un-normalizing the image. This line transforms the pixel values back to the original range (0 to 1) for displaying.
- `npimg = img.numpy()` is converting the image tensor to a numpy array.
- `plt.imshow(np.transpose(npimg, (1, 2, 0)))` is displaying the image. `np.transpose(npimg, (1, 2, 0))` is changing the order of the dimensions from (channels, height, width) to (height, width, channels), which is the format expected by plt.imshow().
- `classes[labels[j]] for j in range(4)` and `classes[predicted[j]] for j in range(4)` print the ground truth labels and the predicted labels for the first four images in the batch.