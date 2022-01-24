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
    # time ranges from 0 to 10
    time = random.uniform(0,10)

    # Each coordinate ranges from -1 to 1
    r_1 = np.array([random.uniform(-1,1), random.uniform(-1,1), random.uniform(-1,1)])
    r_2 = np.array([random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(-1, 1)])
    r_3 = np.array([random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(-1, 1)])

    # Each velocity ranges from -0.2 to 0.2
    v_1 = np.array([random.uniform(-0.2, 0.2), random.uniform(-0.2, 0.2), random.uniform(-0.2, 0.2)])
    v_2 = np.array([random.uniform(-0.2, 0.2), random.uniform(-0.2, 0.2), random.uniform(-0.2, 0.2)])
    v_3 = np.array([random.uniform(-0.2, 0.2), random.uniform(-0.2, 0.2), random.uniform(-0.2, 0.2)])

    # Masses range from 0.8 to 1.2
    m_1 = random.uniform(0.8,1.2)
    m_2 = random.uniform(0.8, 1.2)
    m_3 = random.uniform(0.8, 1.2)

    # Returns our random input vector in this form:
    return np.array([time,
                     r_1,
                     r_2,
                     r_3,
                     v_1,
                     v_2,
                     v_3,
                     m_1,
                     m_2,
                     m_3
        ], dtype='float32')




# Generates input set of given size. Returns torch tensor on appropriate device
def generate_input_set(size):
    result = []
    i = 0
    while i < size:
        result.append(generate_input_vector())
        i += 1
    answer = torch.from_numpy(np.array(result, dtype = 'float32'))
    return answer.to(device)


# Generates target data set based on n by 3d input set whose vectors are of form
# [time,initial_velocity,initial_height]. Takes in torch tensor assumed to be on gpu. Converts it to cpu and
# numpy array for operations. Then returns it back as a torch tensor on appropriate device
def generate_target_set(input_set):
    result = []
    input_set = input_set.cpu().numpy()
    i = 0
    while i < input_set.shape[0]:
        result.append(numerical_solver(input_set[i]))
        i += 1
    answer = torch.from_numpy(np.array([result], dtype = 'float32'))
    return answer.to(device)


# Model takes in 3d input vector. Spits out scalar (height).
# Model has 1000 neurons per hidden layer. Model is moved onto appropriate device
model = to_device(ThreeBodyNet(3, 1000, 1), device)


# Defines the function that trains the model
def fit(epochs, lr, batch_size, model, opt_func=torch.optim.SGD):

    optimizer = opt_func(model.parameters(), lr)

    for epoch in range(epochs):

        # Generates the input and output training data
        inputs = generate_input_set(batch_size)
        targets = generate_target_set(inputs).reshape(batch_size, 1)

        # Trains the model
        loss = model.training_step(inputs, targets)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if epoch % 10000 == 0:
            print(f'Epoch: {epoch}')


# Run for 300,000 epochs.
fit(300000, 1e-5, 100, model)

# Saves the model
torch.save(model.state_dict(), 'Models\FirstModel.pth')

