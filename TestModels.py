import torch
from ThreeBodies import ThreeBodyNet
import numpy as np
from NumericalSolver import get_full_data
import torch
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib as mpl
mpl.rcParams['animation.ffmpeg_path'] = r'D:\Aki\Python37\Lib\site-packages\ffmpeg-5.0-essentials_build\bin\ffmpeg.exe'



imported_model = ThreeBodyNet(22, 1000, 9)
imported_model.load_state_dict(torch.load('Models\ThirdModel.pth'))

def compare_trajectories(w):
    total_time = w[0]
    real_data = get_full_data(w)
    r1_sol = real_data[0]
    r2_sol = real_data[1]
    r3_sol = real_data[2]


    print(r1_sol[-1])
    print(r2_sol[-1])
    print(r3_sol[-1])

    print(imported_model(torch.from_numpy(w))[0:3])
    print(imported_model(torch.from_numpy(w))[3:6])
    print(imported_model(torch.from_numpy(w))[6:9])


    # fig = plt.figure(figsize=(20,20)) #Create 3D axes
    # ax = fig.add_subplot(111,projection="3d") #Plot the orbits
    #
    # real_particle1, = plt.plot([],[], [], color='r')
    # real_particle2, = plt.plot([],[], [], color='g')
    # real_particle3, = plt.plot([],[], [], color='b')
    #
    # model_particle1, = plt.plot([], [], [], color='black')
    # model_particle2, = plt.plot([], [], [], color='yellow')
    # model_particle3, = plt.plot([], [], [], color='orange')
    #
    # def update(i):
    #     t = i/4000 * total_time
    #
    #     real_particle1.set_data(r1_sol[:i, 0], r1_sol[:i, 1])
    #     real_particle1.set_3d_properties(r1_sol[:i,2])
    #     real_particle2.set_data(r2_sol[:i,0], r2_sol[:i,1])
    #     real_particle2.set_3d_properties(r2_sol[:i,2])
    #     real_particle3.set_data(r3_sol[:i, 0], r3_sol[:i, 1])
    #     real_particle3.set_3d_properties(r3_sol[:i, 2])
    #
    #
    #     model_particle1.set_data(imported_model(torch.from_numpy(w)[0]),imported_model(torch.from_numpy(w)[1]))
    #     model_particle1.set_3d_properties(imported_model(torch.from_numpy(w)[2]))
    #     model_particle2.set_data(imported_model(torch.from_numpy(w)[3]), imported_model(torch.from_numpy(w)[4]))
    #     model_particle2.set_3d_properties(imported_model(torch.from_numpy(w)[5]))
    #     model_particle3.set_data(imported_model(torch.from_numpy(w)[6]), imported_model(torch.from_numpy(w)[7]))
    #     model_particle3.set_3d_properties(imported_model(torch.from_numpy(w)[8]))
    #
    #
    #     return real_particle1, real_particle2, real_particle3, model_particle1, model_particle2, model_particle3
    #
    # writer = animation.FFMpegWriter(fps=60)
    # ani = animation.FuncAnimation(fig, update, frames=4000, interval=25, blit = True)
    # ani.save(r"D:\Aki\Pycharm\PycharmProjects\ThreeBodies\sample.mp4", writer = writer)

input = np.array([ 0.0,   -0.13472769, -0.5032093,   0.89470166,  0.85704565,  0.52349514,
  0.19522771, -0.1668427,   0.9241036,   0.5331911,   0.14088221,  0.11630405,
 -0.17256415,  0.06951951,  0.1763274,   0.05063986,  0.15454873,  0.09188691,
  0.02639361,  1.0611788,   1.0217828,   0.9619391 ], dtype="float32")
compare_trajectories(input)