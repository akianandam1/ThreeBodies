import torch
from ThreeBodies import ThreeBodyNet
from NumericalSolver import *
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader


# Function to determine whether gpu is available or not
def get_default_device():
    """Pick GPU if available, else CPU"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')


device = get_default_device()


# Automatically puts data onto the default device
def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)


# Loads the data
input_set = np.load("input_data.npy")
target_set = np.load("target_data.npy")
input_set = torch.from_numpy(input_set).to(device)
target_set = torch.from_numpy(target_set).to(device)
dataset = TensorDataset(input_set, target_set)
batch_size = 1
data_loader = DataLoader(dataset, batch_size, shuffle=True)

# # Initializes and fills the input and target sets
# input_set = np.array([data[0][0]], dtype='float32')
# target_set = np.array([data[0][1]], dtype='float32')
# i = 1
# while i < len(data):
#     input_set = np.append(input_set, np.array([data[i][0]]), axis=0)
#     target_set = np.append(target_set, np.array([data[i][1]]), axis=0)
#     print(i)
#     i += 1
#
# print("Built inputs and outputs")
#
# # Puts them on correct device



# Model takes in 21d input vector. Spits out 9000d output vector (positions).
# Model has 256 neurons per hidden layer. Model is moved onto appropriate device
model = to_device(ThreeBodyNet(21, 256, 9000), device)

# Defines the function that trains the model
def fit(epochs, lr, model, opt_func=torch.optim.Adam):

    # Gets appropriate batch size such that we use all the data by the time we finish our epochs
    optimizer = opt_func(model.parameters(), lr)

    for epoch in range(epochs):
        i = 1
        for inputs, targets in data_loader:
            print(i)
            # # Generates the input and target training data
            # inputs = input_set[(epoch*batch_size):((epoch+1)*batch_size)].to(torch.float32)
            # targets = target_set[(epoch*batch_size):((epoch+1)*batch_size)].to(torch.float32)

            # Trains the model
            loss = model.training_step(inputs, targets)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            i += 1
        print(f"Epoch: {epoch}")


fit(2, 1e-5, model)

# Saves the model
torch.save(model.state_dict(), 'Models\TestModel.pth')
