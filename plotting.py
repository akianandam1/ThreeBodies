from NumericalSolver import get_full_data
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import matplotlib as mpl
import torch
from ThreeBodies import ThreeBodyNet

mpl.rcParams['animation.ffmpeg_path'] = r'D:\Aki\Python37\Lib\site-packages\ffmpeg-5.0-essentials_build\bin\ffmpeg.exe'
imported_model = ThreeBodyNet(21, 256, 9000)
imported_model.load_state_dict(torch.load('Models\TestModel.pth'))

input = np.array([-0.48944658,  0.04835057,  0.11562183,  0.84818554,  0.88853484,
        0.794892  , -0.99460113,  0.6007243 ,  0.16061743,  0.19378561,
        0.08345382, -0.18823653, -0.17162505,  0.10885531,  0.09457611,
       -0.18120962, -0.02358072,  0.19219033,  0.9343363 ,  0.9927892 ,
        1.090716 ], dtype='float32')
sol = get_full_data(input).reshape((1000,9))
r1_sol = sol[:, 0:3]
r2_sol = sol[:, 3:6]
r3_sol = sol[:, 6:9]

model_sol = imported_model(torch.from_numpy(input)).detach().numpy().reshape((1000,9))
model_r1_sol = model_sol[:, 0:3]
model_r2_sol = model_sol[:, 3:6]
model_r3_sol = model_sol[:, 6:9]



# Create figure
fig = plt.figure(figsize=(20, 20))  # Create 3D axes
ax = fig.add_subplot(111, projection="3d")  # Plot the orbits
ax.set_zlim(-3, 3)
ax.set_title("Visualization of orbits of stars in a three body system\n", fontsize=28)

particle1, = plt.plot([], [], [], color='r')
particle2, = plt.plot([], [], [], color='g')
particle3, = plt.plot([], [], [], color='b')
plt.xlim(-3, 3)
plt.ylim(-3, 3)

p1, = plt.plot([], [], marker='o', color='r', label="Real first star")
p2, = plt.plot([], [], marker='o', color='g', label="Real second star")
p3, = plt.plot([], [], marker='o', color='b', label="Real third star")


model_particle1, = plt.plot([], [], [], color='r')
model_particle2, = plt.plot([], [], [], color='g')
model_particle3, = plt.plot([], [], [], color='b')

model_p1, = plt.plot([], [], marker='s', color='r', label="Model's first star")
model_p2, = plt.plot([], [], marker='s', color='g', label="Model's second star")
model_p3, = plt.plot([], [], marker='s', color='b', label="Model's third star")

ax.legend(loc="upper left", fontsize=28)
def update(i):
    particle1.set_data(r1_sol[:i, 0], r1_sol[:i, 1])
    particle1.set_3d_properties(r1_sol[:i, 2])
    particle2.set_data(r2_sol[:i, 0], r2_sol[:i, 1])
    particle2.set_3d_properties(r2_sol[:i, 2])
    particle3.set_data(r3_sol[:i, 0], r3_sol[:i, 1])
    particle3.set_3d_properties(r3_sol[:i, 2])

    p1.set_data(r1_sol[i:i + 1, 0], r1_sol[i:i + 1, 1])
    p1.set_3d_properties(r1_sol[i:i + 1, 2])
    p2.set_data(r2_sol[i:i + 1, 0], r2_sol[i:i + 1, 1])
    p2.set_3d_properties(r2_sol[i:i + 1, 2])
    p3.set_data(r3_sol[i:i + 1, 0], r3_sol[i:i + 1, 1])
    p3.set_3d_properties(r3_sol[i:i + 1, 2])

    model_particle1.set_data(model_r1_sol[:i, 0], model_r1_sol[:i, 1])
    model_particle1.set_3d_properties(model_r1_sol[:i, 2])
    model_particle2.set_data(model_r2_sol[:i, 0], model_r2_sol[:i, 1])
    model_particle2.set_3d_properties(model_r2_sol[:i, 2])
    model_particle3.set_data(model_r3_sol[:i, 0], model_r3_sol[:i, 1])
    model_particle3.set_3d_properties(model_r3_sol[:i, 2])

    model_p1.set_data(model_r1_sol[i:i + 1, 0], model_r1_sol[i:i + 1, 1])
    model_p1.set_3d_properties(model_r1_sol[i:i + 1, 2])
    model_p2.set_data(model_r2_sol[i:i + 1, 0], model_r2_sol[i:i + 1, 1])
    model_p2.set_3d_properties(model_r2_sol[i:i + 1, 2])
    model_p3.set_data(model_r3_sol[i:i + 1, 0], model_r3_sol[i:i + 1, 1])
    model_p3.set_3d_properties(model_r3_sol[i:i + 1, 2])

    return particle1, particle2, particle3, p1, p2, p3, model_particle1, model_particle2, model_particle3, model_p1, model_p2, model_p3


writer = animation.FFMpegWriter(fps=60)
ani = animation.FuncAnimation(fig, update, frames=1000, interval=25, blit=True)
ani.save(r"D:\Aki\Pycharm\PycharmProjects\ThreeBodies\test.mp4", writer=writer)
