import numpy as np
import numpy.random as rand
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from numpy import cos, sin, pi, sqrt
from scipy.integrate import odeint
from scipy.special import erf
import time
import datetime
import os



#CODE INSTRUCTIONS

#Set the parameters below as required. If you wish to save a .mp4 movie and .png plots change plotfigures = savemovie to True (ensure the subdirectories 'Figures/' & 'Movies/' are already made in the directory this programme is saved in).
#Scroll to the bottom where there are calls to seven functions, each simulating a different thing along with instructions about input variables.






#PARAMETRES (set as required)

#physical parameters
G = 1
M = 1 #single galaxy mass
M1 = 1 #double galaxies masses 1 and 2. M1/M2 must be in the range 0.2-->5
M2 = 1
halo_radius = 6 #radius of uniorm density dark matter halo
sigma = 3 #gaussian halo standard deviation (width)
clockwise = True #true = clockwise rotation of galaxies, false = anticlockwise


#Animation/plotting parameters
xy_max = 50 #+/- plotting range
N = 500 #no. integration time steps
animation_length = 10 #movie length (in seconds) - scale animation time step so movie is this long.
t_max = 500
t_selection = [0,50,100,150,200,300,400,499] #time's you wish to plot at (must be integer and < t_max)

#If plotfigures = savemovies = True, the code will save an .mp4 movie and .png plots of the simulation into the Movies/ & Figures/ subdirectories which MUST already exist in the same parent directory as this code is saved.
current_directory = os.path.dirname(__file__)
figure_directory = os.path.join(current_directory, 'Figures/')
movie_directory = os.path.join(current_directory, 'Movies/')
plotfigures = savemovie = False #If False, only an animation will show, no movie file/plots will be saved permanently. options: False/True






#SIMULATION FUNCTIONS
#These are the 5 function types which carry out the integration: ODE, integrate, SIV, GetData & animate. They are explained below.

#ODEs: Returns the derivative to be integrated.
def ODE_single_galaxy(r,t):
    return [r[2],r[3],-G*M*r[0]/(r[0]**2+r[1]**2)**(3/2),-G*M*r[1]/(r[0]**2+r[1]**2)**(3/2)]

def ODE_single_galaxy_halo(r,t):
    Mass = (M*(sqrt(r[0]**2+r[1]**2)/halo_radius)**3 if (sqrt(r[0]**2+r[1]**2)/halo_radius) < 1 else M)
    return [r[2],r[3],-G*Mass*r[0]/(r[0]**2+r[1]**2)**(3/2),-G*Mass*r[1]/(r[0]**2+r[1]**2)**(3/2)]

def ODE_galaxy_motion(R,t): # R = [ x1 , y1 , vx1 , vy1 , x2 , y2 , vx2 , vy2] for galaxy 1 and galaxy 2. returns [vx1, vy1, d(vx1)/dt, d(vy1)/dt, vx2, vy2, d(vx2)/dt, d(vy2)/dt] aka the derivatives
    x , y = R[0]-R[4] , R[1]-R[5]
    return [ R[2] , R[3] , -G*M2*x/(x**2+y**2)**(3/2) , -G*M2*y/(x**2+y**2)**(3/2) , R[6] , R[7] , G*M1*x/(x**2+y**2)**(3/2) , G*M1*y/(x**2+y**2)**(3/2) ]

def ODE_double_galaxy(r,t,R,t_array):
    index = np.searchsorted(t_array,t)
    cgp = [] #current galaxy position
    for i in range(4):
        cgp.append(R[i][index-1])
    x1 , y1 , x2 , y2 = r[0]-cgp[0] , r[1]-cgp[1] , r[0]-cgp[2] , r[1]-cgp[3]
    return [  r[2]  , r[3] ,  -G*M1*x1/(x1**2+y1**2)**(3/2) + -G*M2*x2/(x2**2+y2**2)**(3/2)  ,  -G*M1*y1/(x1**2+y1**2)**(3/2) + -G*M2*y2/(x2**2+y2**2)**(3/2)  ]

def ODE_double_galaxy_halo (r,t,R,t_array):
    index = np.searchsorted(t_array,t)
    cgp = [] #current galaxy position
    for i in range(4):
        cgp.append(R[i][index-1])
    x , y  = [r[0]-cgp[0],r[0]-cgp[2]], [r[1]-cgp[1],r[1]-cgp[3]]
    Mass = [M1,M2]
    for i in range(len(Mass)):
        Mass[i] = (Mass[i]*(sqrt(x[i]**2 + y[i]**2)/halo_radius)**3 if (sqrt(x[i]**2 + y[i]**2)/halo_radius) < 1 else Mass[i])
    return [  r[2]  , r[3] ,  -G*Mass[0]*x[0]/(x[0]**2+y[0]**2)**(3/2) + -G*Mass[1]*x[1]/(x[1]**2+y[1]**2)**(3/2)  ,  -G*Mass[0]*y[0]/(x[0]**2+y[0]**2)**(3/2) + -G*Mass[1]*y[1]/(x[1]**2+y[1]**2)**(3/2)  ]

def ODE_double_galaxy_gaussian_halo (r,t,R,t_array):
    index = np.searchsorted(t_array,t)
    cgp = [] #current galaxy position
    for i in range(4):
        cgp.append(R[i][index-1])
    x , y  = [r[0]-cgp[0],r[0]-cgp[2]], [r[1]-cgp[1],r[1]-cgp[3]]
    Mass = [M1,M2]
    for i in range(len(Mass)):
        Mass[i] = Mass[i]*erf(sqrt(x[i]**2 + y[i]**2)/sigma)
    return [  r[2]  , r[3] ,  -G*Mass[0]*x[0]/(x[0]**2+y[0]**2)**(3/2) + -G*Mass[1]*x[1]/(x[1]**2+y[1]**2)**(3/2)  ,  -G*Mass[0]*y[0]/(x[0]**2+y[0]**2)**(3/2) + -G*Mass[1]*y[1]/(x[1]**2+y[1]**2)**(3/2)  ]



#Functions to call the integrator, scipy.integrate.odeint, and pass them the initial conditions and a t_array to be integrated over.
def integrate_single_galaxy(t,r0):
    r = odeint(ODE_single_galaxy,r0,t)
    return r[:,0],r[:,1]

def integrate_single_galaxy_halo(t,r0):
    r = odeint(ODE_single_galaxy_halo,r0,t)
    return r[:,0],r[:,1]

def integrate_galaxy_motion(t,R0):
    r = odeint(ODE_galaxy_motion,R0,t)
    return r[:,0],r[:,1],r[:,4],r[:,5]

def integrate_double_galaxy(t,r0,R,t_array):
    r = odeint(ODE_double_galaxy,r0,t,args=(R,t_array))
    return r[:,0],r[:,1]

def integrate_double_galaxy_halo(t,r0,R,t_array):
    r = odeint(ODE_double_galaxy_halo ,r0,t,args=(R,t_array))
    return r[:,0],r[:,1]

def integrate_double_galaxy_gaussian_halo(t,r0,R,t_array):
    r = odeint(ODE_double_galaxy_gaussian_halo ,r0,t,args=(R,t_array))
    return r[:,0],r[:,1]


#SIV = Set Initial Values. Returns a vector (or a vector of vectors in the case of multiple stars) with the starting positions and velocites of the stars or galaxies.
def SIV_single_galaxy_ring(r,n): #sets initial conditions for n particles evenly around a circle of radius r. r and n are equal length vectors to allow for multiple rings
    r0 = []
    for j in range(len(n)):
        for i in range(n[j]):
            r0.append([r[j]*cos(i*2*pi/n[j]),r[j]*sin(i*2*pi/n[j]),sqrt(G*M/r[j])*sin(i*2*pi/n[j]),-sqrt(G*M/r[j])*cos(i*2*pi/n[j])])
    return r0

def SIV_single_galaxy_halo_ring(r,n): #sets initial conditions for n particles evenly around a circle of radius r. r and n are equal length vectors to allow for multiple rings
    r0 = []
    for j in range(len(n)):
        Mass = (M*(r[j]/halo_radius)**3 if r[j]/halo_radius < 1 else M)
        for i in range(n[j]):
            r0.append([r[j]*cos(i*2*pi/n[j]),r[j]*sin(i*2*pi/n[j]),sqrt(G*Mass/r[j])*sin(i*2*pi/n[j]),-sqrt(G*Mass/r[j])*cos(i*2*pi/n[j])])
    return r0

def SIV_single_galaxy_random(n,max_r): #sets initial conditions for n with randomly perturbed circular orbits within a max starting radius max_r
    r0 = []
    for i in range(n):
        R = max_r*rand.random()
        theta = 2*pi*rand.random()
        v = sqrt(G*M/R) + 0.1*rand.uniform(-1,1)*sqrt(G*M/R)
        delta = rand.uniform(-pi/6,pi/6)
        r0.append([R*cos(theta),R*sin(theta),v*sin(theta+delta),-v*cos(theta+delta)])
    return r0

def SIV_double_galaxy_ring(r,n,R0): #sets initial conditions for n particles evenly around a circle of radius r about two galaxies with initial conditions defined by R0
    k = (1 if clockwise==True else -1)
    r0 = []
    for j in range(len(n)): #galaxy 1 stars
        for i in range(n[j]):
            r0.append([r[j]*cos(i*2*pi/n[j])+R0[0],r[j]*sin(i*2*pi/n[j])+R0[1],k*sqrt(G*M1/r[j])*sin(i*2*pi/n[j])+R0[2],-k*sqrt(G*M1/r[j])*cos(i*2*pi/n[j])+R0[3]])
    for j in range(len(n)): #galaxy 2 stars
        for i in range(n[j]):
            r0.append([r[j]*cos(i*2*pi/n[j])+R0[4],r[j]*sin(i*2*pi/n[j])+R0[5],k*sqrt(G*M2/r[j])*sin(i*2*pi/n[j])+R0[6],-k*sqrt(G*M2/r[j])*cos(i*2*pi/n[j])+R0[7]])
    return r0

def SIV_double_galaxy_halo_ring(r,n,R0): #sets initial conditions for n particles evenly around a circle of radius r about two galaxies with initial conditions defined by R0
    k = (1 if clockwise==True else -1)
    r0 = []
    for j in range(len(n)): #galaxy 1 stars
        Mass = ((M1*(r[j]/halo_radius)**3) if r[j]/halo_radius < 1 else M)
        for i in range(n[j]):
            r0.append([r[j]*cos(i*2*pi/n[j])+R0[0],r[j]*sin(i*2*pi/n[j])+R0[1],k*sqrt(G*Mass/r[j])*sin(i*2*pi/n[j])+R0[2],-k*sqrt(G*Mass/r[j])*cos(i*2*pi/n[j])+R0[3]])
    for j in range(len(n)): #galaxy 2 stars
        Mass = ((M2*(r[j]/halo_radius)**3) if r[j]/halo_radius < 1 else M)
        for i in range(n[j]):
            r0.append([r[j]*cos(i*2*pi/n[j])+R0[4],r[j]*sin(i*2*pi/n[j])+R0[5],k*sqrt(G*Mass/r[j])*sin(i*2*pi/n[j])+R0[6],-k*sqrt(G*Mass/r[j])*cos(i*2*pi/n[j])+R0[7]])
    return r0

def SIV_double_galaxy_gaussian_halo_ring(r,n,R0): #sets initial conditions for n particles evenly around a circle of radius r about two galaxies with initial conditions defined by R0
    k = (1 if clockwise==True else -1)
    r0 = []
    for j in range(len(n)): #galaxy 1 stars
        Mass = M1*erf(r[j]/sigma)
        for i in range(n[j]):
            r0.append([r[j]*cos(i*2*pi/n[j])+R0[0],r[j]*sin(i*2*pi/n[j])+R0[1],k*sqrt(G*Mass/r[j])*sin(i*2*pi/n[j])+R0[2],-k*sqrt(G*Mass/r[j])*cos(i*2*pi/n[j])+R0[3]])
    for j in range(len(n)): #galaxy 2 stars
        Mass = M1*erf(r[j]/sigma)
        for i in range(n[j]):
            r0.append([r[j]*cos(i*2*pi/n[j])+R0[4],r[j]*sin(i*2*pi/n[j])+R0[5],k*sqrt(G*Mass/r[j])*sin(i*2*pi/n[j])+R0[6],-k*sqrt(G*Mass/r[j])*cos(i*2*pi/n[j])+R0[7]])
    return r0

def SIV_galaxy_motion(s_init,s_closest): #sets initial conditions for the two galaxies, starting at s_init separtion to orbit with zero total energy and pass with closest approach distance s_closest
    h = s_init/(2*sqrt(2))
    theta = theta_for_closest_approach(s_init,s_closest)
    print("Starting angle (deg): %.1f" %(theta*180/pi))
    EffMass = 2*M1*M2/(M1+M2)
    return [-h,h,sqrt(G*EffMass/(h*2*sqrt(2)))*cos(theta),-sqrt(G*EffMass/(h*2*sqrt(2)))*sin(theta),h,-h,-sqrt(G*EffMass/(h*2*sqrt(2)))*cos(theta),sqrt(G*EffMass/(h*2*sqrt(2)))*sin(theta)]



#Given the starting locations and positions of the stars and/or galaxies and passes them, one by one, to the integrator which returns their position throughout
#the simulation as a vector. This function then stacks each of these in a form which can be plotted by the animator and returns these stacked vectors x, y.
#It all (for sanity) prints the percentage of stars completed so the user can guage how long it will take.
def GetData_single_galaxy(r0): #returns a tuple of tensors x,y which contain the x and y positions of all stars at all times
    t = np.linspace(0,t_max,N)
    x, y = integrate_single_galaxy(t,r0[0])
    for i in range(len(r0)-1):
        datax, datay = integrate_single_galaxy(t,r0[i+1])
        x = np.column_stack((x,datax))
        y = np.column_stack((y,datay))
        print("%.1f %% " %(100*i/len(r0)), end="\r")
    return x,y

def GetData_single_galaxy_halo(r0):#returns a tuple of tensors x,y which contain the x and y positions of all stars at all times
    t = np.linspace(0,t_max,N)
    x, y = integrate_single_galaxy_halo(t,r0[0])
    for i in range(len(r0)-1):
        datax, datay = integrate_single_galaxy_halo(t,r0[i+1])
        x = np.column_stack((x,datax))
        y = np.column_stack((y,datay))
        print("%.1f %% " %(100*i/len(r0)), end="\r")
    return x,y

def GetData_galaxy_motion(r0): #returns tuple of vectors x1, y1, x2, y2 for the positions of the galaxies
    t = np.linspace(0,t_max,N)
    x1, y1, x2, y2 = integrate_galaxy_motion(t,r0)
    x = np.column_stack((x1,x2))
    y = np.column_stack((y1,y2))
    return x1, y1, x2, y2

def GetData_double_galaxy(r0,R_galaxies):#returns a tuple of tensors x,y which contain the x and y positions of all stars at all times
    t = np.linspace(0,t_max,N)
    x, y = integrate_double_galaxy(t,r0[0],R_galaxies,t)
    for i in range(len(r0)-1):
        datax, datay = integrate_double_galaxy(t,r0[i+1],R_galaxies,t)
        x = np.column_stack((x,datax))
        y = np.column_stack((y,datay))
        print("%.1f %% " %(100*i/len(r0)), end="\r")
    return x,y

def GetData_double_galaxy_halo(r0,R_galaxies):#returns a tuple of tensors x,y which contain the x and y positions of all stars at all times
    t = np.linspace(0,t_max,N)
    x, y = integrate_double_galaxy_halo(t,r0[0],R_galaxies,t)
    for i in range(len(r0)-1):
        datax, datay = integrate_double_galaxy_halo(t,r0[i+1],R_galaxies,t)
        x = np.column_stack((x,datax))
        y = np.column_stack((y,datay))
        print("%.1f %% " %(100*i/len(r0)), end="\r")
    return x,y

def GetData_double_galaxy_gaussian_halo(r0,R_galaxies):#returns a tuple of tensors x,y which contain the x and y positions of all stars at all times
    t = np.linspace(0,t_max,N)
    x, y = integrate_double_galaxy_gaussian_halo(t,r0[0],R_galaxies,t)
    for i in range(len(r0)-1):
        datax, datay = integrate_double_galaxy_gaussian_halo(t,r0[i+1],R_galaxies,t)
        x = np.column_stack((x,datax))
        y = np.column_stack((y,datay))
        print("%.1f %% " %(100*i/len(r0)), end="\r")
    return x,y




#These animation functions tie everything together so the systems can be plotted easily in a single line call (See bottom).
#They take only the most logical  inputs: radii, number of stars, initial galaxy separation, closest approach etc.
def animation_single_galaxy_ring(Radii,Numbers):
    R0 = SIV_single_galaxy_ring(Radii,Numbers)
    datax, datay = GetData_single_galaxy(R0)
    animate(datax,datay)
    plot_figures(datax,datay)
    return

def animation_single_galaxy_random(r_max,n_total):
    R0 = SIV_single_galaxy_random(n_total,r_max)
    datax, datay = GetData_single_galaxy(R0)
    animate(datax,datay)
    plot_figures(datax,datay)
    return

def animation_single_galaxy_halo_ring(Radii,Numbers):
    R0 = SIV_single_galaxy_halo_ring(Radii,Numbers)
    datax, datay = GetData_single_galaxy_halo(R0)
    animate(datax,datay)
    plot_figures(datax,datay)
    return

def animation_galaxy_motion(s_init,s_closest):
    R0 = SIV_galaxy_motion(s_init,s_closest)
    R_galaxies  = GetData_galaxy_motion(R0)
    datax = np.column_stack((R_galaxies[0],R_galaxies[2]))
    datay = np.column_stack((R_galaxies[1],R_galaxies[3]))
    animate(datax,datay)
    plot_figures(datax,datay)
    return

def animation_double_galaxy(Radii,Numbers,s_init,s_closest):
    R0_galaxies = SIV_galaxy_motion(s_init,s_closest)
    R_galaxies = GetData_galaxy_motion(R0_galaxies)
    R0_stars = SIV_double_galaxy_ring(Radii,Numbers,R0_galaxies)
    datax, datay = GetData_double_galaxy(R0_stars,R_galaxies)
    animate(datax,datay)
    plot_figures(datax,datay)
    return

def animation_double_galaxy_halo(Radii,Numbers,s_init,s_closest):
    R0_galaxies = SIV_galaxy_motion(s_init,s_closest)
    R_galaxies = GetData_galaxy_motion(R0_galaxies)
    R0_stars = SIV_double_galaxy_halo_ring(Radii,Numbers,R0_galaxies)
    datax, datay = GetData_double_galaxy_halo(R0_stars,R_galaxies)
    animate(datax,datay)
    plot_figures(datax,datay)
    return

def animation_double_galaxy_gaussian_halo(Radii,Numbers,s_init,s_closest):
    R0_galaxies = SIV_galaxy_motion(s_init,s_closest)
    R_galaxies = GetData_galaxy_motion(R0_galaxies)
    R0_stars = SIV_double_galaxy_gaussian_halo_ring(Radii,Numbers,R0_galaxies)
    datax, datay = GetData_double_galaxy_gaussian_halo(R0_stars,R_galaxies)
    animate(datax,datay)
    plot_figures(datax,datay)
    return






#OTHER FUNCTIONS

#Returns an angle the two galaxies (initally a distance s_init apart along the y = -x line) must be set of at from the horizontal in order to pass with closest approach s_closest (total energy = 0)
def theta_for_closest_approach(s_init,s_closest):
    Theta = [pi*i/180 for i in range(46)]
    h = s_init/(2*sqrt(2))
    S = []
    for theta in Theta:
        EffMass = 2*M1*M2/(M1+M2)
        initial_galaxy_data = [-h,h,sqrt(G*EffMass/(h*2*sqrt(2)))*cos(theta),-sqrt(G*EffMass/(h*2*sqrt(2)))*sin(theta),h,-h,-sqrt(G*EffMass/(h*2*sqrt(2)))*cos(theta),sqrt(G*EffMass/(h*2*sqrt(2)))*sin(theta)]
        R_gal  = GetData_galaxy_motion(initial_galaxy_data)
        galaxy_separation = sqrt((R_gal[0]-R_gal[2])**2+(R_gal[1]-R_gal[3])**2)
        S.append(abs(s_closest - np.ndarray.min(galaxy_separation)))
    i = S.index(min(S))
    return pi*i/180

#The next two functions animates, and saves, the simulation. Takes x and y as vectors containing the positions of all the particles. eg:  x = [[x1 x2 ... xn]_t=1  [x1 x2 ... xn]_t=2 ... [x1 x2 ... xn]_t=n]
def update_func(num,datax,datay,line):
    line.set_data(datax[num,:],datay[num,:])
    return line,

def animate(datax,datay):
    fig1 = plt.figure()
    line, = plt.plot([],[],".", markersize = 1.5)
    plt.xlim(-xy_max,xy_max)
    plt.ylim(-xy_max,xy_max)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.gca().set_aspect('equal')
    anim = animation.FuncAnimation(fig1, update_func, datax.shape[0], fargs=(datax,datay,line), interval=1000*animation_length/N, blit=True)
    print('Computing time = %.1f s' %(time.time() - T0))
    plt.show()
    if savemovie != False:
        anim.save(movie_directory + '%s.mp4' %datetime.datetime.now().strftime("%I%M%p%B%d"))

#Plots and saves the simulation at all the times given in the t_selection variable at the top.
def plot_figures(datax,datay): #t_selection must be even in length (2,4,6...)
    if plotfigures != False:
        for t in t_selection:
            plt.figure()
            num = int(round((t/t_max)*N))
            plt.plot(datax[num,:],datay[num,:],".", markersize = 1.5)
            plt.xlim(-xy_max,xy_max)
            plt.ylim(-xy_max,xy_max)
            plt.title('Time = %.1f' %t)
            plt.gca().set_aspect('equal')
            plt.xlabel("x")
            plt.ylabel("y")
            plt.savefig(figure_directory + '%s_%g.png' %(datetime.datetime.now().strftime("%I%M%p%B%d"),t), bbox_inches='tight',dpi=500)






#CALL LINES

#Call these one at a time (by unhashing) with the lines below depending on what you would like to simulate. There are 6 in total:

#single_galaxy_ring. A single galaxy of concentric rings of stars around a black hole. Input parametres: ([r0,r1,r2...],[n0,n1,n2...])
#single_galaxy_halo_ring. A single galaxy of concentric rings of stars around a uniform density dark matter halo. Input parametres: ([r0,r1,r2...],[n0,n1,n2...])
#single_galaxy_random: A single galaxy with n_total stars randomly perturbed off circular orbits within a radius r_max. Input parametres: (r_max,n_total)
#galaxy_motion: The motion of a two galaxies (no stars) in parabolic orbit. Input parametres: (starting_seperation,closest_approach)
#double_galaxy: Double galaxy interaction with concentric star rings, using the black hole model. Input parametres: ([r0,r1,r2...],[n0,n1,n2...],starting_seperation,closest_approach,clockwise=True/False) false if galaxies rotate anticlockwise
#double_galaxy_halo: Double galaxy interaction with concentric star rings, using the uniform dark matter halo model. Input parametres: ([r0,r1,r2...],[n0,n1,n2...],starting_seperation,closest_approach,clockwise=True/False)
#double_galaxy_gaussian_halo: Double galaxy interaction with concentric star rings, using the gaussian dark matter halo model. Input parametres: ([r0,r1,r2...],[n0,n1,n2...],starting_seperation,closest_approach,clockwise=True/False)


# animation_double_galaxy_halo([2,3,4,5,6],[24,36,48,60,72],40,10)
# animation_double_galaxy_gaussian_halo([2,3,4,5,6],[24,36,48,60,72],40,10)



T0 = time.time()
# animation_single_galaxy_ring([2,3,4,5,6],[24,36,48,60,72])
# animation_single_galaxy_halo_ring([2,3,4,5,6],[24,36,48,60,72])
# animation_single_galaxy_random(6,240)
# animation_galaxy_motion(40,10)
animation_double_galaxy([2,3,4,5,6],[24,36,48,60,72],40,10)
# animation_double_galaxy_halo([2,3,4,5,6],[24,36,48,60,72],40,10)
# animation_double_galaxy_gaussian_halo([2,3,4,5,6],[24,36,48,60,72],40,10)
