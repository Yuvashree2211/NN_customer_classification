# Developing a Neural Network Classification Model

## AIM

To develop a neural network classification model for the given dataset.

## Problem Statement

An automobile company has plans to enter new markets with their existing products. After intensive market research, they’ve decided that the behavior of the new market is similar to their existing market.

In their existing market, the sales team has classified all customers into 4 segments (A, B, C, D ). Then, they performed segmented outreach and communication for a different segment of customers. This strategy has work exceptionally well for them. They plan to use the same strategy for the new markets.

You are required to help the manager to predict the right group of the new customers.

## Neural Network Model

<img width="973" height="913" alt="image" src="https://github.com/user-attachments/assets/b3a8dd59-594c-4106-8483-369748186754" />


## DESIGN STEPS:


### Step 1: 
Import necessary libraries and load the dataset.

### Step 2: 
Encode categorical variables and normalize numerical features.

### Step 3: 
Split the dataset into training and testing subsets.

### Step 4: 
Design a multi-layer neural network with appropriate activation functions.

### Step 5: 
Train the model using an optimizer and loss function.

### Step 6: 
Evaluate the model and generate a confusion matrix.

### Step 7: 
Use the trained model to classify new data samples.

### Step 8: 
Display the confusion matrix, classification report, and predictions.

## PROGRAM

### Name: YUVASHREE R
### Register Number: 212224040378

```
class PeopleClassifier(nn.Module):
    def __init__(self, input_size, num_classes):
        super(PeopleClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, num_classes)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

```
```
# Initialize the Model, Loss Function, and Optimizer
input_size = X_train.shape[1]
num_classes = 4

model = PeopleClassifier(input_size, num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

```
```
def train_model(model, train_loader, criterion, optimizer, epochs):
    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss:.4f}")
```


## Dataset Information

<img width="947" height="267" alt="image" src="https://github.com/user-attachments/assets/613b3608-07cc-40cf-b54c-a87fdd718020" />




## OUTPUT



### Confusion Matrix
<img width="703" height="579" alt="image" src="https://github.com/user-attachments/assets/c4fd1258-8bed-4009-b027-4f9bd5dfcec2" />


### Classification Report

<img width="545" height="440" alt="image" src="https://github.com/user-attachments/assets/ab12dfc8-8183-497a-9c27-a86d7975d67b" />




### New Sample Data Prediction

<img width="368" height="98" alt="image" src="https://github.com/user-attachments/assets/a7b1e15b-6a02-4a62-a467-b2669ccc7c59" />


## RESULT

Thus the neural network classification model was successfully developed.
