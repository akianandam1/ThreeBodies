from Solutions import r1_sol, r2_sol, r3_sol
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib as mpl
mpl.rcParams['animation.ffmpeg_path'] = r'D:\MAIN\Python37\Lib\site-packages\ffmpeg-5.0-essentials_build\bin\ffmpeg.exe'

#Create figure
fig = plt.figure(figsize=(20,20)) #Create 3D axes
ax = fig.add_subplot(111,projection="3d") #Plot the orbits

particle1, = plt.plot([],[], [], color='r')
particle2, = plt.plot([],[], [], color='g')
particle3, = plt.plot([],[], [], color='b')

# p1, = ax.scatter(r1_sol[0,0], r1_sol[0,1], r1_sol[0,2], marker = 'o', color='red')
# p2, = ax.scatter(r2_sol[0,0], r2_sol[0,1], r2_sol[0,2], marker = 'o', color='green')
# p3, = ax.scatter(r3_sol[0,0], r2_sol[0,1], r2_sol[0,2], marker = 'o', color='blue')

def update(i):
    particle1.set_data(r1_sol[:i, 0], r1_sol[:i, 1])
    particle1.set_3d_properties(r1_sol[:i,2])
    particle2.set_data(r2_sol[:i,0], r2_sol[:i,1])
    particle2.set_3d_properties(r2_sol[:i,2])
    particle3.set_data(r3_sol[:i, 0], r3_sol[:i, 1])
    particle3.set_3d_properties(r3_sol[:i, 2])

    # p1.set_data(r1_sol[i+1, 0], r1_sol[i+1, 1])
    # p1.set_3d_properties(r1_sol[i+1, 2])
    # p2.set_data(r2_sol[i+1, 0], r2_sol[i+1, 1])
    # p2.set_3d_properties(r2_sol[i+1, 2])
    # p3.set_data(r3_sol[i+1, 0], r3_sol[i+1, 1])
    # p3.set_3d_properties(r3_sol[i+1, 2])
    return particle1, particle2, particle3 #, p1, p2, p3


writer = animation.FFMpegWriter(fps=60)
ani = animation.FuncAnimation(fig, update, frames=4000, interval=25, blit = True)
ani.save(r"D:\MAIN\PycharmProjects\ThreeBodies\sample.mp4", writer = writer)


