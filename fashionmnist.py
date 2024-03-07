import torch
import torchvision
import torchvision.transforms.v2 as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import numpy as np

from matplotlib import pyplot as plt

# Set seeds for reproducability
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(0)


'''
Preprocessing the fashion mnist dataset.
'''

# Defining the transformation: pixel array to tensor, normalizes around grayscale standards
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

# Size of batches
batch_size = 64

'''
Load the dataseting and creating train/test splits
'''

trainset = torchvision.datasets.FashionMNIST(root = './data', train = True, transform = transform, download = True)
trainloader = torch.utils.data.DataLoader(trainset, batch_size = batch_size, shuffle = True)

testset = torchvision.datasets.FashionMNIST(root = './data', train = False, transform = transform, download = True)
testloader = torch.utils.data.DataLoader(testset, batch_size = batch_size, shuffle = False)

'''
Purely feed-forward neural network
'''

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        
        self.flatten = nn.Flatten()
        
        self.input_layer = nn.Linear(in_features=784, out_features=256)
        self.hidden1 = nn.Linear(in_features=256, out_features=128)
        self.hidden2 = nn.Linear(in_features=128, out_features=64)
        self.output_layer = nn.Linear(in_features=64, out_features=10)

        self.drop = nn.Dropout(.2)
        self.relu = nn.ReLU()

    def forward(self, x):

        x = self.flatten(x)
        x = self.input_layer(x)
        x = self.relu(x)
        x = self.drop(x)
        x = self.hidden1(x)
        x = self.relu(x)
        x = self.drop(x)
        x = self.hidden2(x)
        x = self.relu(x)
        x = self.drop(x)
        x = self.output_layer(x)
        
        return x

# Instantiate the neural network   
net = Net()

'''
Defining the loss function (Cross Entropy) and Optimizer (Adam)
'''

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr = 0.001)

'''
Training the model
'''

num_epochs = 5
training_loss = []

for epoch in range(num_epochs):  # loop through the epochs
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # Get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward + backprop + optimization
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    training_loss.append(running_loss/len(trainloader))
    print(f"Training loss: {running_loss / len(trainloader)}")

print('Finished Training')


'''
Model Evaluation
'''

correct = 0
total = 0

with torch.no_grad():
    for data in testloader:
        images, labels = data  # Get the inputs
        outputs = net(images)  # Forward pass to get outputs
        _, predicted = torch.max(outputs.data, 1)  # Take class with highest energy as prediction
        total += labels.size(0)  # Increase the total count
        correct += (predicted == labels).sum().item()  # Increase the correct count

accuracy = 100 * correct / total
print('Accuracy: ', correct/total)

'''
Plotting loss and sample classification extraction
'''

# Plot training loss over epochs
plt.figure(figsize=(10, 6))
plt.plot(range(1, num_epochs + 1), training_loss, label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Over Time')
plt.legend()
plt.savefig('training_loss_plot.png')

# Extract correctly and falsely identified image

correctly_classified_img = None
incorrectly_classified_img = None
true_label_correct = None
true_label_incorrect = None
predicted_label_correct = None
predicted_label_incorrect = None

with torch.no_grad():
    for data in testloader:
        images, labels = data
        output = net(images)
        _, predicted = torch.max(outputs.data, 1)
        
        # Iterates through current batch
        for i in range(labels.size(0)):
            # Checks if label equal to predicted label and no correct image has been found yet
            if labels[i] == predicted[i] and correctly_classified_img is None:
                # Saves index of correctly classified image
                correctly_classified_img = images[i]
                # Saves correctly classified label
                true_label_correct = labels[i].item()
                # Saves predicted label 
                predicted_label_correct = predicted[i].item()
            # Checks if label is not equal to predicted (incorrectly classified)
            elif labels[i] != predicted[i] and incorrectly_classified_img is None:
                # Saves index of incorrectly classified image
                incorrectly_classified_img = images[i]
                # Stores true label of incorrectly classified image
                true_label_incorrect = labels[i].item()
                # Stores predicted label of incorrectly classified image
                predicted_label_incorrect = predicted[i].item()
                
            # Breaks inner for loop once both images have been found
            if correctly_classified_img is not None and incorrectly_classified_img is not None:
                break
            
        # Breaks outer for loop once both images have been found
        if correctly_classified_img is not None and incorrectly_classified_img is not None:
            break

# Fashion MNIST Label Dictionary

mnist_labels = {0: 'T-shirt', 1: 'Trouser', 2: 'Pullover', 3: 'Dress', 4: 'Coat', 5: 'Sandal', 6: 'Shirt', 7: 'Sneaker', 8: 'Bag', 9: 'Ankle boot'}

# Function to show image
def show_image(img, label, predicted_label, title):
    plt.clf()
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.title(f"{title}\nTrue label: {label}\nPredicted label: {predicted_label}")
    plt.savefig(title + '.png')
    plt.show()
    
# Plot correctly classified image
show_image(correctly_classified_img, mnist_labels[true_label_correct], mnist_labels[predicted_label_correct], "Correctly Classified Image")

# Plot incorrectly classified image
show_image(incorrectly_classified_img, mnist_labels[true_label_incorrect], mnist_labels[predicted_label_incorrect], "Incorrectly Classified Image")
