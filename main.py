import torch
import numpy as np
from ThreeBodies import ThreeBodyNet
import random
from NumericalSolver import *

# Function to determine whether gpu is available or not
def get_default_device():
    """Pick GPU if available, else CPU"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')


device = get_default_device()


#Automatically puts data onto the default device
def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)


# Generates a random input vector of np array form [time, initial velocity, initial height]
def generate_input_vector():
    return np.array([
        random.uniform(0, 5), # time

        random.uniform(-1, 1),
        random.uniform(-1, 1), # particle 1's position
        random.uniform(-1, 1),

        random.uniform(-1, 1),
        random.uniform(-1, 1), # particle 2's position
        random.uniform(-1, 1),

        random.uniform(-1, 1),
        random.uniform(-1, 1), # particle 3's position
        random.uniform(-1, 1),

        random.uniform(-0.2, 0.2),
        random.uniform(-0.2, 0.2), # particle 1's velocity
        random.uniform(-0.2, 0.2),

        random.uniform(-0.2, 0.2),
        random.uniform(-0.2, 0.2), # particle 2's velocity
        random.uniform(-0.2, 0.2),

        random.uniform(-0.2, 0.2),
        random.uniform(-0.2, 0.2), # particle 3's velocity
        random.uniform(-0.2, 0.2),

        random.uniform(0.8, 1.2),
        random.uniform(0.8, 1.2), # masses of each particle
        random.uniform(0.8, 1.2)

    ], dtype = 'float32')



# Generates input set of given size. Returns torch tensor on appropriate device
def generate_input_set(size):
    result = np.array([generate_input_vector()], dtype = "float32")
    i = 1
    while i < size:
        result = np.append(result, np.array([generate_input_vector()], dtype = "float32"), axis=0)
        i += 1
    answer = torch.from_numpy(result)

    return answer.to(device)


# Generates target data set based on n by 3d input set whose vectors are of form
# [time,initial_velocity,initial_height]. Takes in torch tensor assumed to be on gpu. Converts it to cpu and
# numpy array for operations. Then returns it back as a torch tensor on appropriate device
def generate_target_set(input_set):
    input_set = input_set.cpu().numpy()
    result = np.array([numerical_solver(input_set[0])], dtype='float32')
    i = 1
    while i < input_set.shape[0]:
        result = np.append(result, np.array([numerical_solver(input_set[i])], dtype = "float32"), axis=0)
        i += 1
    answer = torch.from_numpy(result)
    return answer.to(device)



# Model takes in 22d input vector. Spits out 9d output vector (positions).
# Model has 1000 neurons per hidden layer. Model is moved onto appropriate device
model = to_device(ThreeBodyNet(22, 1000, 9), device)


# Defines the function that trains the model
def fit(epochs, lr, batch_size, model, opt_func=torch.optim.SGD):

    optimizer = opt_func(model.parameters(), lr)

    for epoch in range(epochs):
        try:
            # Generates the input and output training data
            inputs = generate_input_set(batch_size)
            targets = generate_target_set(inputs)

            # Trains the model
            loss = model.training_step(inputs, targets)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            print(f"Epoch: {epoch}")
        except:
            pass


# Run for 300,000 epochs.
fit(10000, 1e-5, 50, model)

# Saves the model
torch.save(model.state_dict(), 'Models\ThirdModel.pth')

