from NumericalSolver import get_full_data
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import matplotlib as mpl
from ModelSolver import get_model_data

mpl.rcParams['animation.ffmpeg_path'] = r'D:\Aki\Python37\Lib\site-packages\ffmpeg-5.0-essentials_build\bin\ffmpeg.exe'


input = np.array([10, -0.13472769, -0.5032093, 0.89470166, 0.85704565, 0.52349514,
                  0.19522771, -0.1668427, 0.9241036, 0.5331911, 0.14088221, 0.11630405,
                  -0.17256415, 0.06951951, 0.1763274, 0.05063986, 0.15454873, 0.09188691,
                  0.02639361, 1.0611788, 1.0217828, 0.9619391], dtype="float32")
r1_sol = get_full_data(input)[0]
r2_sol = get_full_data(input)[1]
r3_sol = get_full_data(input)[2]

model_r1_sol = get_model_data(input)[:, 0:3]
model_r2_sol = get_model_data(input)[:, 3:6]
model_r3_sol = get_model_data(input)[:, 6:9]


# Create figure
fig = plt.figure(figsize=(20, 20))  # Create 3D axes
ax = fig.add_subplot(111, projection="3d")  # Plot the orbits
ax.set_zlim(-15, 15)
ax.set_title("Visualization of orbits of stars in a three body system\n", fontsize=28)

particle1, = plt.plot([], [], [], color='r')
particle2, = plt.plot([], [], [], color='g')
particle3, = plt.plot([], [], [], color='b')
plt.xlim(-15, 15)
plt.ylim(-15, 15)

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
ani = animation.FuncAnimation(fig, update, frames=4000, interval=25, blit=True)
ani.save(r"D:\Aki\Pycharm\PycharmProjects\ThreeBodies\sample.mp4", writer=writer)
