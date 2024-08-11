# Reccurent Neural Networks
Recurrent Neural Network (RNN) is a type of Neural Network where the output from the previous step is fed as input to the current step. In traditional neural networks, all the inputs and outputs are independent of each other. The main and most important feature of RNN is its hidden state, which remembers some information about a sequence. The state is also referred to as the Memory State since it remembers the previous input to the network. It uses the same parameters for each input as it performs the same task on all the inputs or hidden layers to produce the output (GeeksForGeeks, 2024).
![RNN](https://miro.medium.com/v2/resize:fit:1200/1*HgAY1lLMYSANqtgTgwWeXQ.png)
[Source](https://miro.medium.com/v2/resize:fit:1200/1*HgAY1lLMYSANqtgTgwWeXQ.png)

# Long Short-Term Memory Networks
LSTMs have a more complex structure that includes gates (input, forget, and output gates) that control the flow of information, allowing them to learn what to keep or forget.
![LSTM](https://i.sstatic.net/RHNrZ.jpg)
[Source](https://i.sstatic.net/RHNrZ.jpg)

# About
The Daily Minimum Temperatures in Melbourne dataset contains the daily minimum temperatures in Melbourne, Australia, from January 1st, 1981, to December 31st, 1990. The dataset has two columns, which are respectively the date and the minimum temperature associated with that day.

# Data preprocessing
- Load the dataset into a dataframe.
- Convert the first column into datetime objects.
- Rename the dataframe columns.
- Remove non-digit characters in the temperature column.
- Set the date as the index.
- Scale the data into a range between 0 and 1 to ensure that all values are on a similar scale, which helps the model converge more effectively during training.
- Divide the data into sequences where each sequence `x` consists of 12 time steps (months), and the corresponding target `y` is the next time step.
- Convert the sequences and labels to PyTorch tensors.

# Model creation
- Split the data into training and test sets (80/20).
- Create an LSTM model that has one LSTM layer and a fully connected layer.
- Initialize the initial hidden state and cell state of the LSTM with zeros and store them in the self.hidden_Cell attribute.
- The forward method is the main computation function of the LSTM class. It takes an input sequence (input_seq) as an argument. Inside the method, the input sequence is reshaped using the view method to match the expected input shape of the LSTM layer. The shape is (sequence_length, batch_size, input_size), where sequence_length is the length of the input sequence, batch_size is set to 1, and input_size is the number of features in the input sequence.
- Pass the reshaped input sequence through the LSTM layer (self.lstm). Return the output of the LSTM layer (lstm_out) and the updated hidden state and cell state (self.hidden_cell).
- Reshape the LSTM output again using the view method to match the expected input shape of the linear layer. The shape is (sequence_length, output_size).
- Finally, pass the reshaped LSTM output through the linear layer (self.linear) to obtain the predictions. Use the -1 index to select the last prediction in the sequence, and then return it.

# Model training
- Use the Mean Squared Error (MSE) to measure the difference between predicted and actual values.
- Use the Adam optimizer to adjust the learning rate dynamically, making the training process efficient.
- Train the model for 100 epochs (full passes through the training data). For each sequence, compare the model's predictions to the actual labels, and backpropagate the error to update the model's weights.

# Model evaluation
- Evaluate the model's performance on the test data without updating weights (model.eval() ensures this). Compare the predictions with the actual values.
- Calculate the Root Mean Squared Error to measure the model's prediction accuracy. Lower values indicate better performance.
- Plot the actual vs. predicted values to visualize how well the model performed.

# Predicting future values
Use the trained model to predict future temperature values (for the next 2 years / 24 months) based on the last seq_length time steps.

# References
[GeeksForGeeks](https://www.geeksforgeeks.org/introduction-to-recurrent-neural-network/)