# file_name.py
# Python 3.6
"""
Author: Jacob Marglous
Created: Wed Apr 29 18:41:27 2020
Modified: Wed Apr 29 18:41:27 2020

Description
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as am
from pylab import *

#@jit(nopython = True)
def changepositions(AApositions, chainlength):
    '''
    Performs a new self-avoiding random walk from a residue selected at random.
    '''
    pivotpoint = np.random.randint(0, chainlength)
    
    AApositions_new = np.copy(AApositions)
    
    x = [AApositions_new[0, pivotpoint, 0]]
    y = [AApositions_new[0, pivotpoint, 1]]
    bonds = []
  
    #print("starting point of step is ", AApositions_new[0, pivotpoint, 0], AApositions_new[0, pivotpoint, 1])
    positions = set()
    for i in range(pivotpoint+1):
        positions.add((AApositions_new[0, i, 0], AApositions_new[0, i, 1]))
    
    #print(pivotpoint)
    #print(positions)
    stuck = 0
    
    for i in range(chainlength-pivotpoint-1):
        deltas = [(1,0), (0,1), (-1,0), (0,-1)]
        deltas_feasible = []  #deltas_feasible stores the available directions 
        for dx, dy in deltas:
            if (x[-1] + dx, y[-1] + dy) not in positions:  #checks if direction leads to a site notvisited before
                deltas_feasible.append((dx,dy))
        if deltas_feasible:  #checks if there is a direction available
            dx, dy = deltas_feasible[np.random.randint(0,len(deltas_feasible))]  #choose a direction at random among available ones
            positions.add((x[-1] + dx, y[-1] + dy))
            x.append(x[-1] + dx)
            y.append(y[-1] + dy)
            if ((dx, dy) == (1,0)):
                bonds.append(1)
            if ((dx, dy) == (0, 1)):
                bonds.append(2)
            if ((dx, dy) == (-1, 0)):
                bonds.append(3)
            if ((dx, dy) == (0, -1)):
                bonds.append(4)
        
        else:  #in that case the walk is stuck
            stuck = 1
            steps = i+1
            return AApositions, AApositions
            break
            #terminate the walk prematurely
        #steps = chainlength-1
        
    np.array(x)
    np.array(y)
    bonds.append(0)
    np.array(bonds)
    
    #print(x)
    #print(y)
    
    for i in range(chainlength-pivotpoint):
        AApositions_new[0, pivotpoint+i, 0] = x[i]
        AApositions_new[0, pivotpoint+i, 1] = y[i]
        AApositions_new[0, pivotpoint+i, 2] = bonds[i]
    
    #print(pivotpoint)

    return AApositions, AApositions_new

def stateLattice(L, chainlength, AApositions):
    '''
    Locates positions of hydrophobic residues on entire lattice
    '''
    state = np.zeros((L, L), int)
    for i in range(chainlength):
        state[int(AApositions[0, i, 0]), int(AApositions[0, i,1])] = AApositions[0, i,3]
    
    return state

#@jit(nopython = True)
def totalEnergy(state, L):
        '''
        Sums interactions on state lattice to calculate total energy. Divides sum by two because each interaction
        is double counted (once for [i, j] and once for [j, i]).
        '''
        E_list = [] 
        L1 = L-1
        for i in range(L):
            for j in range(L):
                E = -state[i, j] * (state[(i+1)%L, j] + state[(i+L1)%L, j] + state[i, (j+1)%L] + state[i, (j+L1)%L])
                E_list.append(E)
        np.array(E_list)
        totalEnergy = np.sum(E_list)/2

        return totalEnergy

#@jit(nopython = True)
def momentOfInertia (AApositions, chainlength):
    squaredradii = []
        
    for i in range(chainlength):
        ri = (AApositions[0, i, 0]-10)**2 + (AApositions[0, i, 1]-10)**2
        squaredradii.append(ri)
        
    np.array(squaredradii)
        
    momentOfInertia = np.sum(squaredradii)
        
    return momentOfInertia

#@jit(nopython = True)
def MCstep_jit(L, chainlength, T, AApositions, acceptedMoves, energy):
    
    AApositions, AApositions_new = changepositions(AApositions, chainlength) #make Monte Carlo step by performing self-avoiding random walk from random res. on protein
    
    #Generate state of lattice for old and new protein configuration.
    state_AApositions = stateLattice(L, chainlength, AApositions)
    state_AApositions_new = stateLattice(L, chainlength, AApositions_new)
    
    #Calculate energy of lattice for old and new protein configuration.
    energy_AApositions = totalEnergy(state_AApositions, L)
    energy_AApositions_new = totalEnergy(state_AApositions_new, L)
    
    #Calculate the difference in lattice energy of the two states.
    dE = energy_AApositions_new - energy_AApositions
    
    #Generate a random value to which dE will be compared.
    randomValue = np.random.random()
    if dE <= 0 or np.exp(-dE/T) > randomValue:
        acceptedMoves += 1
        AApositions = AApositions_new
        state = state_AApositions_new
        energy += dE
    
    AA_positions_momentOfInertia = momentOfInertia(AApositions, chainlength)
    
    return AApositions, acceptedMoves, energy, momentOfInertia
    
class Ising2D (object):
    
    def __init__(self, L, chainlength, hydrophobicity, temperature):
        
        self.L = L
        #self.N = L**2
        
        self.chainlength = chainlength
        
        self.hydrophobicity = hydrophobicity
        
        self.temperature = temperature
        
        self.AApositions = self.initialPositions(self.L, self.chainlength, self.hydrophobicity)
        
        self.state = self.stateLattice(self.L, self.chainlength, self.AApositions)
        
        self.energy = self.totalEnergy(self.state, self.L)
        
        self.reset()
        
        self.momentOfInertia = self.MOI(self.AApositions, self.chainlength)
    
    def initialPositions(self, L, chainlength, hydrophobicity):
    
        res1pos = int(L/2)
    
        x, y, bonds = [res1pos], [res1pos], [0]
        positions = set([(res1pos,res1pos)])  #positions is a set that stores all sites visited by the walk
        stuck = 0
        for i in range(chainlength-1):
            deltas = [(1,0), (0,1), (-1,0), (0,-1)]
            deltas_feasible = []  #deltas_feasible stores the available directions 
            for dx, dy in deltas:
                if (x[-1] + dx, y[-1] + dy) not in positions:  #checks if direction leads to a site notvisited before
                    deltas_feasible.append((dx,dy))
            if deltas_feasible:  #checks if there is a direction available
                dx, dy = deltas_feasible[np.random.randint(0,len(deltas_feasible))]  #choose a direction at random among available ones
                positions.add((x[-1] + dx, y[-1] + dy))
                x.append(x[-1] + dx)
                y.append(y[-1] + dy)
                if ((dx, dy) == (1,0)):
                    bonds.append(1)
                if ((dx, dy) == (0, 1)):
                    bonds.append(2)
                if ((dx, dy) == (-1, 0)):
                    bonds.append(3)
                if ((dx, dy) == (0, -1)):
                    bonds.append(4)

            else:  #in that case the walk is stuck
                stuck = 1
                steps = i+1
                break  #terminate the walk prematurely
            steps = chainlength-1
        
        np.array(x)
        np.array(y)
        bonds.pop(0)
        bonds.append(0)
        np.array(bonds)
    
        AApositions = np.dstack([x, y, bonds, hydrophobicity])
    
        return AApositions
    
    def stateLattice(self, L, chainlength, AApositions):
        '''
        Locates positions of hydrophobic residues on entire lattice
        '''
        state = np.zeros((self.L, self.L), int)
        for i in range(chainlength):
            state[int(AApositions[0, i, 0]), int(AApositions[0, i,1])] = AApositions[0, i,3]
        return state
    
    def totalEnergy(self, state, L):
        '''
        Sums interactions on identity lattice to calculate total energy. Divides sum by two because each interaction
        is double counted (once for [i, j] and once for [j, i]).
        '''
        E_list = [] 
        L1 = L-1
        for i in range(L):
            for j in range(L):
                E = -state[i, j] * (state[(i+1)%L, j] + state[(i+L1)%L, j] + state[i, (j+1)%L] + state[i, (j+L1)%L])
                E_list.append(E)
        np.array(E_list)
        totalEnergy = np.sum(E_list)/2

        return totalEnergy
    
    def MOI (self, AApositions, chainlength):
        x_avg = np.average(AApositions[0, :, 0])
        y_avg = np.average(AApositions[0, :, 1])
        centerofmass = np.array([x_avg, y_avg])
        
        squaredradii = []
        
        for i in range(chainlength):
            ri = (AApositions[0, i, 0]-centerofmass[0])**2 + (AApositions[0, i, 1]-centerofmass[1])**2
            squaredradii.append(ri)
        np.array(squaredradii)
        
        momentOfInertia = np.sum(squaredradii)
        
        return momentOfInertia
    
    def increment_T(self, T_increment, reset = True):
        
        T_new = self.temperature + T_increment
        
        if T_new <= 0:
            T_new = self.temperature
            
        self.temperature = T_new
        if reset:
            self.reset()
    
    def reset(self):
        
        self.monteCarloSteps = 0
        self.acceptedMoves = 0
        self.energyArray = np.array([], int)
        self.momentOfInertiaArray = np.array([], int)
        
    def monteCarloStep(self):
        
        L = self.L
        chainlength = self.chainlength
        T = self.temperature
        AApositions = self.AApositions
        acceptedMoves = self.acceptedMoves
        energy = self.energy 
    
        AApositions, acceptedMoves, energy, momentOfInertia = MCstep_jit(L, chainlength, T, AApositions, acceptedMoves, energy)
        
        self.AApositions = AApositions
        #self.state = state
        self.acceptedMoves = acceptedMoves
        self.energy = energy
        self.momentOfInertia = self.MOI(self.AApositions, self.chainlength)
        
        self.energyArray.append(self.energy)
        self.momentOfInertiaArray.append(self.momentOfInertia)
        self.monteCarloSteps += 1
        
    def steps(self, number = 100):
        
        self.energyArray = self.energyArray.tolist()
        self.momentOfInertiaArray = self.momentOfInertiaArray.tolist()
        
        for k in range(number):
            self.monteCarloStep()
                
        self.energyArray = np.asarray(self.energyArray)
        self.momentOfInertiaArray = np.asarray(self.momentOfInertiaArray)
            
    ##Observables
    def specificHeat(self):
        return (self.energyArray.std() / self.temperature)**2/(self.L**2)
    
    def plot_positions(self, save = False):
        plt.scatter(self.AApositions[0,:, 0], self.AApositions[0,:,1], c = self.AApositions[0, :, 3])
        plt.plot(self.AApositions[0,:, 0], self.AApositions[0,:,1])
        if save:
            plt.savefig("Positions.png")
            
    def plot_totalEnergy(self, save = False):
        steps_array = np.arange(self.monteCarloSteps)
        plt.plot(steps_array, self.energyArray)
        plt.xlabel("Monte Carlo Steps")
        plt.ylabel("Conformational Energy")
        plt.title("Energy vs. Folding Time")
        if save:
            plt.savefig("StepsVsTotalEnergy.png")
            
    def plot_momentOfInertia(self, save = False):
        steps_array = np.arange(self.monteCarloSteps)
        plt.plot(steps_array, self.momentOfInertiaArray)
        plt.xlabel("Monte Carlo Steps")
        plt.ylabel("Moment Of Inertia")
        plt.title("Structure vs. Folding Time")
        if save:
            plt.savefig("StepsVsMomentOfInertia.png")

    def basicFig(self, save = False):
        fig = plt.figure(figsize = [6, 12])
    
        ax1 = fig.add_subplot(221)
        ax2 = fig.add_subplot(222)
        ax3 = fig.add_subplot(223)
        ax4 = fig.add_subplot(224)
        
        ax1.scatter(self.AApositions[0,:, 0], self.AApositions[0,:,1], c = self.AApositions[0, :, 3])
        ax1.plot(self.AApositions[0,:, 0], self.AApositions[0,:,1])
        ax1.axis("equal")
        ax1.set_title("Initial Conformation")
        self.reset()
        self.steps(1000)
    
        ax2.scatter(self.AApositions[0,:, 0], self.AApositions[0,:,1], c = self.AApositions[0, :, 3])
        ax2.plot(self.AApositions[0,:, 0], self.AApositions[0,:,1])
        ax2.axis("equal")
        ax2.set_title("Final Conformation")
        steps_array = np.arange(self.monteCarloSteps)
        
        ax3.plot(steps_array, self.energyArray)
        ax3.set_xlabel("Monte Carlo Steps")
        ax3.set_ylabel("Conformational Energy")
        ax3.set_title("Energy vs. Folding Time")
    
        ax4.plot(steps_array, self.momentOfInertiaArray)
        ax4.set_xlabel("Monte Carlo Steps")
        ax4.set_ylabel("Moment Of Inertia")
        ax4.set_title("Structure vs. Folding Time")
        
        if save:
            plt.savefig("StepsVsMomentOfInertia.png")
        
        plt.show()
        
     ##Some simulated annealing methods
    def cool(self, T_max, T_min, tau = 1e4):
        """
        Method to apply exponential cooling
        """
        t = 0
        T = T_max
        energy_array = []
        T_array = []
    
        #energy = deepcopy(cool_protein.energy)
    
        while T > T_min:
            t += 1
            T = T_max*np.exp(-t/tau)
        
            self.temperature = T
            self.steps(1)
        
            energy_array.append(self.energy)
            T_array.append(self.temperature)
    
        return self.AApositions, energy_array, T_array
    
    def anneal(self, cycles, T_max = .2, T_min = .001, tau = 100):
        """
        Run multiple rapid heat and slow cool cycles to find global minimum.
        """
    
        AApositions, energy_array, T_array = self.cool(T_max, T_min, tau)
    
        for k in range(cycles):
            AApositions_test, energy_test, T_test = self.cool(T_max, T_min, tau)
        
            print("Current min energy: {:f}\t Trial min energy: {:f}".format(energy_array[-1], energy_test[-1]))
        
            if energy_test[-1] < energy_array[-1]:
                energy_array[:] = energy_test[:]
                AApositions[:, :, :] = AApositions_test[:, :, :]
            
        fig = plt.figure(figsize = [6, 12])
        ax = fig.add_subplot(111)
        ax.plot(T_array, energy_array)
        ax.set_title("Simulated Annealing")
        ax.set_xlabel("Temperature")
        ax.set_ylabel("Conformational Energy")
        plt.show()
        
        return AApositions, energy_array, T_array
    
        
AA_10 = np.array([1,1,0,0, 1, 1, 0, 0, 0, 1])
AA_20 = np.array([0,1,1,1,0,0,0,0,0,1,0,0,1,0,0,0,1,0,0,1])

def anim_fold(L = 40, chainlength = 20, hydrophobicity = AA_20, temperature=.1, saveanim = False, fname = "animation.mp4"):
        
    protein = Ising2D(L = L, chainlength = chainlength, hydrophobicity = hydrophobicity, temperature = temperature)

    def updatefig(i):
        
        protein.steps(number = 1)
        
        xy = np.array([protein.AApositions[0, :, 0], protein.AApositions[0, :, 1]])
        plts[0].set_offsets(xy.T)
        #ax1.set_xlim([5, 15])
        #ax1.set_ylim([5, 15])
        
        #T_text.set_text("Temperature = {:0.3f}".format(protein.temperature))
        
        plts[1].set_data(protein.AApositions[0, :, 0], protein.AApositions[0, :, 1])
        
        plts[2].set_data(np.arange(protein.monteCarloSteps), protein.energyArray)
        ax2.set_xlim([0, protein.monteCarloSteps+10])
    
        plts[3].set_data(np.arange(protein.monteCarloSteps), protein.momentOfInertiaArray)
        ax3.set_xlim([0, protein.monteCarloSteps+10])
        ax3.set_ylim([0, 250])                      
        return plts, T_text
                                  
    #cmap = cm.get_cmap('Set3')
    
    fig = plt.figure(figsize = [6, 12])
    ax1 = fig.add_subplot(311)
    ax2 = fig.add_subplot(312)
    ax3 = fig.add_subplot(313)
    
    ax2.set_xlabel("Monte Carlo Steps", fontsize = 16)
    ax2.set_ylabel("Conformational Energy", fontsize = 16)
    ax2.set_ylim([-10.1, .1])
    ax2.set_xlim([0, 100])
                              
    
    ax3.set_xlabel("Monte Carlo Steps", fontsize = 16)
    ax3.set_ylabel("Moment of Inertia", fontsize = 16)
    ax3.set_ylim([-.1, 200.1])
    ax3.set_xlim([0, 100])
                              
    fig.tight_layout()
                              
    protein.steps(number = 1)
    
    config = ax1.scatter(protein.AApositions[0, :, 0], protein.AApositions[0, :, 1], c = protein.AApositions[0, :, 3])
    ax1.set_xlim([chainlength-10, chainlength+10])
    ax1.set_ylim([chainlength-10, chainlength+10])
    links, = ax1.plot(protein.AApositions[0,:, 0], protein.AApositions[0,:,1])
                              
    EnergyTime, = ax2.plot(protein.energyArray, lw = 2)
    
    MomentOfInertiaTime, = ax3.plot(protein.momentOfInertiaArray, lw = 2)
                              
    plts = [config, links, EnergyTime, MomentOfInertiaTime]
    
    T_text = ax1.text(.05, .92, " ", transform = ax1.transAxes, fontsize = 16, color = 'k')
    
    ani = am.FuncAnimation(fig, updatefig, frames = 5000, interval = 10, blit = False)
    
    plt.show()
    
    if saveanim:
        ani.savefig(fname, fps = 120)

    else:
        plt.show()
        
    return ani, protein
    
         

