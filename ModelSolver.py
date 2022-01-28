from ThreeBodies import ThreeBodyNet
import numpy as np
import torch

np.set_printoptions(threshold=np.inf)
imported_model = ThreeBodyNet(22, 1000, 9)
imported_model.load_state_dict(torch.load('Models\ThirdModel.pth'))

def get_model_data(input_vector):
    time = input_vector[0]
    i = 0
    input_vector[0] = i
    data = np.array([imported_model(torch.from_numpy(input_vector)).detach().numpy()])
    i += 0.001
    while i <= time:
        input_vector[0] = i
        data = np.append(data, np.array([imported_model(torch.from_numpy(input_vector)).detach().numpy()]), axis=0)
        i += 0.001
    return data

