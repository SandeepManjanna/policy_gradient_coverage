#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import math
import sys
import pickle
import time

reward_map_size = 40
pad_size = reward_map_size-1
world_map_size = reward_map_size + 2*(pad_size)
curr_r_map_size = reward_map_size + pad_size
curr_r_pad = (curr_r_map_size-1)/2
num_actions = 4
num_features = int(sys.argv[2])#4#curr_r_map_size**2
param_len = num_features*num_actions

num_iterations = 500
num_trajectories = 20
Tau_horizon = 400#int(0.9*reward_map_size**2)
is_action_valid = True
rand_start_pos = True
plot = True
fileNm = "totreward_5x5_200_base"

#Discount factor gamma
gamma = 0.9
#Learning rate
Eta = float(sys.argv[3])#0.0015

def makeFig():
    plt.plot(xList,traj_reward_list)
    plt.plot(xList,max_reward_list)
    plt.plot(xList,path_max_reward_list)

#phi_prime for grid-wise (100x100) features with no feature aggregation
def get_phi_prime_39601_feat(worldmap,curr_pos):
    curr_r_map = worldmap[curr_pos[1]-curr_r_pad:curr_pos[1]+curr_r_pad+1,curr_pos[0]-curr_r_pad:curr_pos[0]+curr_r_pad+1]
    phi_prime=np.reshape(curr_r_map,(1,curr_r_map.size))
    return phi_prime

#phi_prime for grid-wise (50x50) features with no feature aggregation
def get_phi_prime_9801_feat(worldmap,curr_pos):
    curr_r_map = worldmap[curr_pos[1]-curr_r_pad:curr_pos[1]+curr_r_pad+1,curr_pos[0]-curr_r_pad:curr_pos[0]+curr_r_pad+1]
    phi_prime=np.reshape(curr_r_map,(1,curr_r_map.size))
    return phi_prime


#phi_prime for grid-wise (15x15) features with no feature aggregation
def get_phi_prime_841_feat(worldmap,curr_pos):
    #start = time.time()
    curr_r_map = worldmap[curr_pos[1]-curr_r_pad:curr_pos[1]+curr_r_pad+1,curr_pos[0]-curr_r_pad:curr_pos[0]+curr_r_pad+1]
    phi_prime=np.reshape(curr_r_map,(1,curr_r_map.size))
    #end = time.time()
    #print (end-start)
    return phi_prime

#phi_prime for grid-wise features with no feature aggregation
def get_phi_prime_289_feat(worldmap,curr_pos):
    #start = time.time()
    curr_r_map = worldmap[curr_pos[1]-curr_r_pad:curr_pos[1]+curr_r_pad+1,curr_pos[0]-curr_r_pad:curr_pos[0]+curr_r_pad+1]
    phi_prime=np.reshape(curr_r_map,(1,curr_r_map.size))
    #end = time.time()
    #print (end-start)
    return phi_prime

#phi_prime for grid-wise features with no feature aggregation
def get_phi_prime_169_feat(worldmap,curr_pos):
    #start = time.time()
    curr_r_map = worldmap[curr_pos[1]-curr_r_pad:curr_pos[1]+curr_r_pad+1,curr_pos[0]-curr_r_pad:curr_pos[0]+curr_r_pad+1]
    phi_prime=np.reshape(curr_r_map,(1,curr_r_map.size))
    #end = time.time()
    #print (end-start)
    return phi_prime

#phi_prime for 4-feature aggregation --> above_row, below_row, before_column, after_column
def get_phi_prime_4_feat(worldmap,curr_pos):
    #start = time.time()
    curr_r_map = worldmap[curr_pos[1]-curr_r_pad:curr_pos[1]+curr_r_pad+1,curr_pos[0]-curr_r_pad:curr_pos[0]+curr_r_pad+1]
    avg = np.count_nonzero(curr_r_map)
    phi_prime = []
    temp = np.copy(curr_r_map[:,0:curr_r_pad])
    avg = np.count_nonzero(temp)
    if(avg==0):avg=1
    phi_prime.append(np.sum(temp)/avg)
    temp = np.copy(curr_r_map[0:curr_r_pad,:])
    avg = np.count_nonzero(temp)
    if(avg==0):avg=1
    phi_prime.append(np.sum(temp)/avg)
    temp = np.copy(curr_r_map[:,curr_r_pad+1:])
    avg = np.count_nonzero(temp)
    if(avg==0):avg=1
    phi_prime.append(np.sum(temp)/avg)
    temp = np.copy(curr_r_map[curr_r_pad+1:,:])
    avg = np.count_nonzero(temp)
    if(avg==0):avg=1
    phi_prime.append(np.sum(temp)/avg)
    phi_prime = np.array([phi_prime])
    #end = time.time()
    #print (end-start)
    return phi_prime

#phi_prime for 8-feature aggregation --> 
def get_phi_prime_8_feat(worldmap,curr_pos):
    #start = time.time()
    curr_r_map = worldmap[curr_pos[1]-curr_r_pad:curr_pos[1]+curr_r_pad+1,curr_pos[0]-curr_r_pad:curr_pos[0]+curr_r_pad+1]
    avg = float(curr_r_pad)
    phi_prime = []
    phi_prime.append(np.sum(curr_r_map[curr_r_pad,0:curr_r_pad])/avg)
    phi_prime.append(np.sum(curr_r_map[0:curr_r_pad,0:curr_r_pad])/(avg**2))
    phi_prime.append(np.sum(curr_r_map[0:curr_r_pad,curr_r_pad])/avg)
    phi_prime.append(np.sum(curr_r_map[0:curr_r_pad,curr_r_pad+1:])/(avg**2))
    phi_prime.append(np.sum(curr_r_map[curr_r_pad,curr_r_pad+1:])/avg)
    phi_prime.append(np.sum(curr_r_map[curr_r_pad+1:,curr_r_pad+1:])/(avg**2))
    phi_prime.append(np.sum(curr_r_map[curr_r_pad+1:,curr_r_pad])/avg)
    phi_prime.append(np.sum(curr_r_map[curr_r_pad+1:,0:curr_r_pad])/(avg**2))
    phi_prime = np.array([phi_prime])
    #end = time.time()
    #print (end-start)
    return phi_prime

#phi_prime for 24-feature aggregation --> High resolution to low resolution
def get_phi_prime_24_feat(worldmap,curr_pos):
    wmap = np.copy(worldmap)
    r=curr_pos[1]
    c=curr_pos[0]
    phi_prime = []
    phi_prime.append(wmap[r,c-1])
    phi_prime.append(wmap[r-1,c-1])
    phi_prime.append(wmap[r-1,c])
    phi_prime.append(wmap[r-1,c+1])
    phi_prime.append(wmap[r,c+1])
    phi_prime.append(wmap[r+1,c+1])
    phi_prime.append(wmap[r+1,c])
    phi_prime.append(wmap[r+1,c-1])
    
    temp = np.copy(wmap[r-1:r-1+3,c-4:c-4+3])
    avg = np.size(temp)
    if(avg==0):avg=1
    phi_prime.append(np.sum(temp)/avg)
    temp = np.copy(wmap[r-4:r-4+3,c-4:c-4+3])
    avg = np.size(temp)
    if(avg==0):avg=1
    phi_prime.append(np.sum(temp)/avg)
    temp = np.copy(wmap[r-4:r-4+3,c-1:c-1+3])
    avg = np.size(temp)
    if(avg==0):avg=1
    phi_prime.append(np.sum(temp)/avg)
    temp = np.copy(wmap[r-4:r-4+3,c+2:c+2+3])
    avg = np.size(temp)
    if(avg==0):avg=1
    phi_prime.append(np.sum(temp)/avg)
    temp = np.copy(wmap[r-1:r-1+3,c+2:c+2+3])
    avg = np.size(temp)
    if(avg==0):avg=1
    phi_prime.append(np.sum(temp)/avg)
    temp = np.copy(wmap[r+2:r+2+3,c+2:c+2+3])
    avg = np.size(temp)
    if(avg==0):avg=1
    phi_prime.append(np.sum(temp)/avg)
    temp = np.copy(wmap[r+2:r+2+3,c-1:c-1+3])
    avg = np.size(temp)
    if(avg==0):avg=1
    phi_prime.append(np.sum(temp)/avg)
    temp = np.copy(wmap[r+2:r+2+3,c-4:c-4+3])
    avg = np.size(temp)
    if(avg==0):avg=1
    phi_prime.append(np.sum(temp)/avg)
    
    temp = np.copy(wmap[r-4:r-4+9,0:c-4])
    avg = np.size(temp)
    if(avg==0):avg=1
    phi_prime.append(np.sum(temp)/avg)
    temp = np.copy(wmap[0:r-4,0:c-4])
    avg = np.size(temp)
    if(avg==0):avg=1
    phi_prime.append(np.sum(temp)/avg)
    temp = np.copy(wmap[0:r-4,c-4:c-4+9])
    avg = np.size(temp)
    if(avg==0):avg=1
    phi_prime.append(np.sum(temp)/avg)
    temp = np.copy(wmap[0:r-4,c-4+9:])
    avg = np.size(temp)
    if(avg==0):avg=1
    phi_prime.append(np.sum(temp)/avg)
    temp = np.copy(wmap[r-4:r-4+9,c-4+9:])
    avg = np.size(temp)
    if(avg==0):avg=1
    phi_prime.append(np.sum(temp)/avg)
    temp = np.copy(wmap[r-4+9:,c-4+9:])
    avg = np.size(temp)
    if(avg==0):avg=1
    phi_prime.append(np.sum(temp)/avg)
    temp = np.copy(wmap[r-4+9:,c-4:c-4+9])
    avg = np.size(temp)
    if(avg==0):avg=1
    phi_prime.append(np.sum(temp)/avg)
    temp = np.copy(wmap[r-4+9:,0:c-4])
    avg = np.size(temp)
    if(avg==0):avg=1
    phi_prime.append(np.sum(temp)/avg)
    
    phi_prime = np.array([phi_prime])
    return phi_prime
   
#phi_prime for 24-feature uniform aggregation --> 
def get_phi_prime_24_feat_1(worldmap,curr_pos):
    curr_r_map = worldmap[curr_pos[1]-curr_r_pad:curr_pos[1]+curr_r_pad+1,curr_pos[0]-curr_r_pad:curr_pos[0]+curr_r_pad+1]
    avg = float(curr_r_pad)/2
    phi_prime = []
    phi_prime.append(np.sum(curr_r_map[curr_r_pad,(curr_r_pad/2):curr_r_pad])/avg) #1
    phi_prime.append(np.sum(curr_r_map[(curr_r_pad/2):curr_r_pad,(curr_r_pad/2):curr_r_pad])/(avg**2)) #2
    phi_prime.append(np.sum(curr_r_map[(curr_r_pad/2):curr_r_pad,curr_r_pad])/avg) #3
    phi_prime.append(np.sum(curr_r_map[(curr_r_pad/2):curr_r_pad,curr_r_pad+1:curr_r_pad+1+(curr_r_pad/2)])/(avg**2)) #4
    phi_prime.append(np.sum(curr_r_map[curr_r_pad,curr_r_pad+1:curr_r_pad+1+(curr_r_pad/2)])/avg) #5
    phi_prime.append(np.sum(curr_r_map[curr_r_pad+1:curr_r_pad+1+(curr_r_pad/2),curr_r_pad+1:curr_r_pad+1+(curr_r_pad/2)])/(avg**2)) #6
    phi_prime.append(np.sum(curr_r_map[curr_r_pad+1:curr_r_pad+1+(curr_r_pad/2),curr_r_pad])/avg) #7
    phi_prime.append(np.sum(curr_r_map[curr_r_pad+1:curr_r_pad+1+(curr_r_pad/2),(curr_r_pad/2):curr_r_pad])/(avg**2)) #8

    phi_prime.append(np.sum(curr_r_map[curr_r_pad,0:(curr_r_pad/2)])/avg) #9
    phi_prime.append(np.sum(curr_r_map[(curr_r_pad/2):curr_r_pad,0:(curr_r_pad/2)])/(avg**2)) #10
    phi_prime.append(np.sum(curr_r_map[0:(curr_r_pad/2),0:(curr_r_pad/2)])/(avg**2)) #11
    phi_prime.append(np.sum(curr_r_map[0:(curr_r_pad/2),(curr_r_pad/2):curr_r_pad])/(avg**2)) #12
    phi_prime.append(np.sum(curr_r_map[0:(curr_r_pad/2),curr_r_pad])/avg) #13
    phi_prime.append(np.sum(curr_r_map[0:(curr_r_pad/2),curr_r_pad+1:curr_r_pad+1+(curr_r_pad/2)])/(avg**2)) #14
    phi_prime.append(np.sum(curr_r_map[0:(curr_r_pad/2),curr_r_pad+1+(curr_r_pad/2):])/(avg**2)) #15
    phi_prime.append(np.sum(curr_r_map[(curr_r_pad/2):curr_r_pad,curr_r_pad+1+(curr_r_pad/2):])/(avg**2)) #16
    phi_prime.append(np.sum(curr_r_map[curr_r_pad,curr_r_pad+1+(curr_r_pad/2):])/(avg**2)) #17
    phi_prime.append(np.sum(curr_r_map[curr_r_pad+1:curr_r_pad+1+(curr_r_pad/2),curr_r_pad+1+(curr_r_pad/2):])/(avg**2)) #18
    phi_prime.append(np.sum(curr_r_map[curr_r_pad+1+(curr_r_pad/2):,curr_r_pad+1+(curr_r_pad/2):])/(avg**2)) #19
    phi_prime.append(np.sum(curr_r_map[curr_r_pad+1+(curr_r_pad/2):,curr_r_pad+1:curr_r_pad+1+(curr_r_pad/2)])/(avg**2)) #20
    phi_prime.append(np.sum(curr_r_map[curr_r_pad+1+(curr_r_pad/2):,curr_r_pad])/avg) #21
    phi_prime.append(np.sum(curr_r_map[curr_r_pad+1+(curr_r_pad/2):,(curr_r_pad/2):curr_r_pad])/(avg**2)) #22
    phi_prime.append(np.sum(curr_r_map[curr_r_pad+1+(curr_r_pad/2):,0:(curr_r_pad/2)])/(avg**2)) #23
    phi_prime.append(np.sum(curr_r_map[curr_r_pad+1:curr_r_pad+1+(curr_r_pad/2),0:(curr_r_pad/2)])/(avg**2)) #24

    phi_prime = np.array([phi_prime])
    #end = time.time()
    #print (end-start)
    return phi_prime
 
#Defining a dictionary for functions
funcdict = {'get_phi_prime_841_feat':get_phi_prime_841_feat, 'get_phi_prime_289_feat':get_phi_prime_289_feat, 'get_phi_prime_169_feat':get_phi_prime_169_feat, 'get_phi_prime_4_feat':get_phi_prime_4_feat, 'get_phi_prime_8_feat':get_phi_prime_8_feat, 'get_phi_prime_24_feat':get_phi_prime_24_feat, 'get_phi_prime_39601_feat':get_phi_prime_39601_feat, 'get_phi_prime_9801_feat':get_phi_prime_9801_feat}

#get_phi_prime = ''
get_phi_prime = 'get_phi_prime_'+str(num_features)+'_feat'

def get_phi(worldmap,curr_pos,curr_action):
    #Actions: 0 - LEFT, 1 - UP, 2 - RIGHT, and 3 - DOWN
    identity_left = np.zeros(num_features)
    identity_up = np.zeros(num_features)
    identity_right = np.zeros(num_features)
    identity_down = np.zeros(num_features)

    if curr_action == 0:
        identity_left = np.ones(num_features)
    elif curr_action == 1:
        identity_up = np.ones(num_features)
    elif curr_action == 2:
        identity_right = np.ones(num_features)
    else:
        identity_down = np.ones(num_features)

    identity = np.concatenate((identity_left,identity_up,identity_right,identity_down))
    phi_prime = funcdict[get_phi_prime](worldmap,curr_pos)
    phi = np.tile(phi_prime,num_actions)
    phi = np.multiply(phi,identity)
    phi = np.transpose(phi)
    return phi

def get_pi(worldmap,pos,act,theta,isPrint=False):
    phi_left = get_phi(worldmap,pos,0)
    phi_up = get_phi(worldmap,pos,1)
    phi_right = get_phi(worldmap,pos,2)
    phi_down = get_phi(worldmap,pos,3)

    dot_l = np.dot(np.transpose(theta),phi_left)[0,0]
    dot_u = np.dot(np.transpose(theta),phi_up)[0,0]
    dot_r = np.dot(np.transpose(theta),phi_right)[0,0]
    dot_d = np.dot(np.transpose(theta),phi_down)[0,0]


    #Making sure the range for numpy.exp
    dot_max = np.max([dot_l,dot_u,dot_r,dot_d])
    dot_l = dot_l - dot_max
    dot_u = dot_u - dot_max
    dot_r = dot_r - dot_max
    dot_d = dot_d - dot_max
    
    #if (np.isinf(dot_l) or np.isinf(dot_u) or np.isinf(dot_r) or np.isinf(dot_d)):
    if isPrint:
        print dot_l
        print dot_u
        print dot_r
        print dot_d
        print theta
        sys.exit()

    exp_l = np.exp(dot_l)
    exp_u = np.exp(dot_u)
    exp_r = np.exp(dot_r)
    exp_d = np.exp(dot_d)
    exp_sum = exp_l + exp_u + exp_r + exp_d

    phi_act = get_phi(worldmap,pos,act)
    dot_act = np.dot(np.transpose(theta),phi_act)
    dot_act = dot_act - dot_max
    exp_act = np.exp(dot_act)

    return (exp_act/exp_sum)

def sample_action(worldmap,pos,theta,isprint=False,maxPolicy=False):
    phi_left = get_phi(worldmap,pos,0)
    phi_up = get_phi(worldmap,pos,1)
    phi_right = get_phi(worldmap,pos,2)
    phi_down = get_phi(worldmap,pos,3)

    dot_l = np.dot(np.transpose(theta),phi_left)[0,0]
    dot_u = np.dot(np.transpose(theta),phi_up)[0,0]
    dot_r = np.dot(np.transpose(theta),phi_right)[0,0]
    dot_d = np.dot(np.transpose(theta),phi_down)[0,0]

    #Making sure the range for numpy.exp
    dot_max = np.max([dot_l,dot_u,dot_r,dot_d])
    dot_l = dot_l - dot_max
    dot_u = dot_u - dot_max
    dot_r = dot_r - dot_max
    dot_d = dot_d - dot_max
    
    exp_l = np.exp(dot_l)
    exp_u = np.exp(dot_u)
    exp_r = np.exp(dot_r)
    exp_d = np.exp(dot_d)
    exp_sum = exp_l + exp_u + exp_r + exp_d

    prob_l = exp_l / exp_sum
    prob_u = exp_u / exp_sum
    prob_r = exp_r / exp_sum
    prob_d = exp_d / exp_sum

    if isprint:
        print "Position:"+str(pos)
        print prob_l
        print prob_u
        print prob_r
        print prob_d

    if maxPolicy:
        next_action = np.argmax([prob_l,prob_u,prob_r,prob_d])
    else:
        next_action = np.random.choice(num_actions,1,p=[prob_l,prob_u,prob_r,prob_d])[0]
    #print next_action
    return next_action

def get_next_state(worldmap,curr_pos,curr_action):
    global is_action_valid
    next_pos = list(curr_pos)
    if(curr_action == 0):
        if(curr_pos[0]-(curr_r_pad+1) > -1):
            next_pos[0] = curr_pos[0]-1
        else:
            is_action_valid = False
    elif(curr_action == 1):
        if(curr_pos[1]-(curr_r_pad+1) > -1):
            next_pos[1] = curr_pos[1]-1
        else:
            is_action_valid = False
    elif(curr_action == 2):
        if(curr_pos[0]+(curr_r_pad+1) < worldmap.shape[1]):
            next_pos[0] = curr_pos[0]+1
        else:
            is_action_valid = False
    elif(curr_action == 3):
        if(curr_pos[1]+(curr_r_pad+1) < worldmap.shape[0]):
            next_pos[1] = curr_pos[1]+1
        else:
            is_action_valid = False

    return next_pos

class Trajectory:
    def __init__(self,curr_pos,curr_act,curr_reward):
        self.curr_pos=list(curr_pos)
        self.curr_act=curr_act
        self.curr_reward=curr_reward
    def get_curr_pos(self):
        return self.curr_pos
    def get_curr_act(self):
        return self.curr_act
    def get_curr_reward(self):
        return self.curr_reward
    def __str__(self):
        return str(self.curr_pos)+" "+str(self.curr_act)+" "+str(self.curr_reward)

def generate_trajectories(num_trajectories,worldmap,curr_pos,theta,isPrint=False,maxPolicy=False,rand_start=True):
    #Array of trajectories starting from current position.
    global is_action_valid
    copy_worldmap = np.copy(worldmap)
    Tau = np.ndarray(shape=(num_trajectories,Tau_horizon), dtype=object)
    start_pos = list(curr_pos)
    for i in range(num_trajectories):
        p=[]
        for prob in range(reward_map_size):
            p.append(1.0/reward_map_size)
        pos1 = np.random.choice(range(reward_map_size),1,p)[0]+pad_size
        pos2 = np.random.choice(range(reward_map_size),1,p)[0]+pad_size
        curr_pos = np.array([pos1,pos2])
        if not rand_start:
            curr_pos = []
            curr_pos = list(start_pos)
        worldmap = None
        worldmap = np.copy(copy_worldmap)
        for j in range(Tau_horizon):
            curr_action = sample_action(worldmap,curr_pos,theta,isPrint,maxPolicy)
            next_pos = get_next_state(worldmap,curr_pos,curr_action)
            if is_action_valid:
                curr_reward = worldmap[next_pos[1],next_pos[0]]
                worldmap[curr_pos[1],curr_pos[0]] = 0
            else:
                curr_reward = -2
                is_action_valid = True
            t = Trajectory(curr_pos,curr_action,curr_reward)
            Tau[i][j] = t
            curr_pos = next_pos
    return Tau

def get_derivative(Tau,worldmap,theta):
    pos = Tau.curr_pos
    act = Tau.curr_act
    phi = get_phi(worldmap,pos,act)
    sum_b = 0
    for b in range(num_actions):
        sum_b = sum_b + (get_pi(worldmap,pos,b,theta) * get_phi(worldmap,pos,b))
    delta = phi - sum_b
    return delta

def plot_theta(theta):
    theta_reshape = np.reshape(theta,(num_actions,num_features))
    if np.sqrt(num_features).is_integer():
        theta_l = np.reshape(theta_reshape[0],(np.sqrt(num_features),np.sqrt(num_features)))
        theta_u = np.reshape(theta_reshape[1],(np.sqrt(num_features),np.sqrt(num_features)))
        theta_r = np.reshape(theta_reshape[2],(np.sqrt(num_features),np.sqrt(num_features)))
        theta_d = np.reshape(theta_reshape[3],(np.sqrt(num_features),np.sqrt(num_features)))
    else:
        theta_l = np.reshape(theta_reshape[0],(2, num_features/2))
        theta_u = np.reshape(theta_reshape[1],(2, num_features/2))
        theta_r = np.reshape(theta_reshape[2],(2, num_features/2))
        theta_d = np.reshape(theta_reshape[3],(2, num_features/2))
    plt.ion()
    fig1 = plt.figure()
    plt.subplot(221)
    plt.imshow(theta_l, interpolation='nearest', cmap=plt.cm.viridis)
    plt.clim(-1,1.5)
    plt.colorbar()
    plt.title('Theta_Left')

    plt.subplot(222)
    plt.imshow(theta_u, interpolation='nearest', cmap=plt.cm.viridis)
    plt.clim(-1,1.5)
    plt.colorbar()
    plt.title('Theta_Up')

    plt.subplot(223)
    plt.imshow(theta_r, interpolation='nearest', cmap=plt.cm.viridis)
    plt.clim(-1,1.5)
    plt.colorbar()
    plt.title('Theta_Right')

    plt.subplot(224)
    plt.imshow(theta_d, interpolation='nearest', cmap=plt.cm.viridis)
    plt.clim(-1,1.5)
    plt.colorbar()
    plt.title('Theta_Down')

    plt.show(block=True)

def get_maximum_path_reward(worldmap,curr_pos,theta,path_max_reward_list,discount_reward_list):
    num_traj=1
    max_path_reward = 0
    discount_reward = 0
    Tau = generate_trajectories(num_traj,worldmap,curr_pos,theta,maxPolicy=True,rand_start=False)
    for i in range(num_traj):
        #traj_max_reward = 0
        for j in range(Tau_horizon):
            #print Tau[i][j]
            max_path_reward = max_path_reward + Tau[i][j].curr_reward
            discount_reward = discount_reward + (gamma**j)*Tau[i][j].curr_reward 
        #    traj_max_reward = traj_max_reward + Tau[i][j].curr_reward
        #print 'The trajectory wise max reward = '+str(traj_max_reward)
    max_path_reward = max_path_reward / num_traj
    discount_reward = discount_reward / num_traj
    print 'The Maximum reward with trajectory = '+str(max_path_reward)
    print 'The discounted reward = '+str(discount_reward)
    path_max_reward_list.append(max_path_reward)
    discount_reward_list.append(discount_reward)

#CURRENT POSITION : curr_pos = [x,y] = [no_cols,no_rows]
#Actions: 0 - LEFT, 1 - UP, 2 - RIGHT, and 3 - DOWN

if __name__ == '__main__':
    
    #global num_features, Eta, get_phi_prime
    #num_features = int(sys.argv[2])
    #Eta = float(sys.argv[3])
    #get_phi_prime = 'get_phi_prime_'+str(num_features)+'_feat'
    
    #rewardmap = np.array([[1,1,5],[1,1,1],[1,4,10]])
    #rewardmap = np.array([[1,1,5,1,1],[1,1,13,4,4],[1,4,10,1,1],[1,1,1,1,1],[1,1,5,5,5]])
    #rewardmap = np.array([[15,1,1,1,1],[1,1,1,1,15],[1,1,1,1,1],[1,1,1,1,1],[1,10,10,1,1]])
    #rewardmap = np.array([[1,1,5,2,2,1,1],[1,1,13,4,4,1,1],[1,4,10,1,1,15,10],[1,1,1,1,1,5,4],[1,1,5,5,5,1,3],[1,1,11,1,1,3,3],[2,3,10,11,1,5,1]])
    #rewardmap = np.array([[1,1,5,2,2,1,1,1,1],[1,1,13,4,4,1,1,1,1],[1,4,10,1,1,15,10,1,1],[1,1,1,1,1,5,4,1,1],[1,1,5,5,5,1,3,1,1],[1,1,11,1,1,3,3,1,1],[2,3,10,11,1,5,1,1,1],[1,1,13,4,4,1,1,1,1],[1,1,11,1,1,3,3,1,1]])

    #Reward map 50x50
    rewardmap = np.ones((50,50))
    rewardmap[45:50,47:50] = 12
    rewardmap[20:25,32:36] = 5
    rewardmap[0:4,46:50] = 10
    rewardmap[39:42,3:8] = 15
    rewardmap[0:3,0:5] = 6
    rewardmap[11:13,6:8] = 3
    
    '''
    rewardmap = np.ones((100,100))
    rewardmap[90:100,95:100] = 10
    rewardmap[0:5,0:10] = 5
    rewardmap[0:15,0:3] = 7
    rewardmap[95:100,0:15] = 3
    rewardmap[20:24,3:9] = 15
    rewardmap[56:60,71:73] = 12
    rewardmap[88:91,44:50] = 6

    
    rewardmap = np.array([[ 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 5.0, 5.0, 5.0, 1.0, 1.0, 10.0, 5.0, 2.0],
         [ 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 5.0, 5.0, 5.0, 1.0, 1.0, 10.0, 5.0, 3.0],
         [ 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 5.0, 5.0, 15.0, 1.0, 1.0, 10.0, 4.0, 1.0],
         [ 15.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
         [ 15.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
         [ 15.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
         [ 1.0, 1.0, 7.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
         [ 1.0, 1.0, 9.0, 1.0, 1.0, 1.0, 1.0, 10.0, 12.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
         [ 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 11.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
         [ 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
         [ 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
         [ 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
         [ 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 10.0, 10.0, 10.0],
         [ 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 10.0, 10.0, 10.0],
         [ 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 10.0, 10.0, 10.0]])
    '''

    rewardmap = pickle.load(open('/home/msandeep/workspaces/iser2018/train_set.pkl',"r"))
    rewardmap = 10 * rewardmap
    maximum_reward = sum(-np.sort(-np.reshape(rewardmap,(1,reward_map_size**2))[0])[0:Tau_horizon])
    worldmap = np.zeros((world_map_size,world_map_size))
    worldmap[pad_size:pad_size+reward_map_size,pad_size:pad_size+reward_map_size] = rewardmap
    curr_pos = np.array([reward_map_size,reward_map_size])

    theta = np.random.rand(param_len,1)*0.1
    print worldmap
    plt.ion()
    #fig = plt.figure()
    xList=list()
    traj_reward_list=list()
    max_reward_list=list()
    path_max_reward_list=list()
    discount_reward_list=list()
    tot_time = 0

    orig_worldmap = np.copy(worldmap)

    for iterations in range(num_iterations):
        start = time.time()
        worldmap = None
        worldmap = np.copy(orig_worldmap)
        Tau = generate_trajectories(num_trajectories,worldmap,curr_pos,theta,rand_start=rand_start_pos)
        g_T = 0
        tot_reward=0
        r_str = ""
        sum_R_t = []
        sum_R_t = np.zeros(Tau_horizon)
        for i in range(num_trajectories):
            g_Tau = 0
            traj_reward = 0
            worldmap = None
            worldmap = np.copy(orig_worldmap)
            for j in range(Tau_horizon):
                #Rolling out each of the trajectories
                R_t = 0
                #Total reward in a trajectory
                tot_reward += Tau[i][j].curr_reward
                traj_reward += Tau[i][j].curr_reward
                for t in range(j,Tau_horizon): 
                    R_t = R_t + gamma**(t-j)*Tau[i][t].curr_reward
                sum_R_t[j] = sum_R_t[j] + R_t
                A_t = R_t - (sum_R_t[j]/(i+1))
                worldmap[Tau[i][j].curr_pos[1],Tau[i][j].curr_pos[0]] = 0
                g_t = get_derivative(Tau[i][j],worldmap,theta) * A_t
                g_Tau = g_Tau + g_t
            if iterations%10==0:
                r_str = r_str + str(traj_reward) + " "
            g_T = g_T + g_Tau
        g_T = g_T / num_trajectories
        #print "THE max and min of Theta = "
        #print np.max(theta)
        #print np.min(theta)
        g_T = g_T / num_features
        tot_reward = tot_reward / num_trajectories
        theta = theta + Eta*g_T
        #plotting the total reward vs. no. of iterations
        if plot==True and iterations%10==0:
            print tot_reward
            xList.append(iterations)
            traj_reward_list.append(tot_reward)
            max_reward_list.append(maximum_reward)
            curr_pos1 = np.array([reward_map_size,reward_map_size])
            worldmap1 = np.copy(orig_worldmap)
            theta1 = np.copy(theta)
            get_maximum_path_reward(worldmap1,curr_pos1,theta1,path_max_reward_list,discount_reward_list)
            makeFig()
            plt.draw()
            plt.pause(0.00001)
        if iterations%101 == 0:
            pickle.dump([rewardmap, gamma, Eta, num_trajectories, Tau_horizon, num_iterations, theta, Tau, xList, traj_reward_list, max_reward_list, path_max_reward_list, discount_reward_list, (float(tot_time)/num_iterations)],open('./reef_train_40x40_100_1.pkl',"w"))
        '''
        if plot==True and iterations == num_iterations-1:
            xList.append(iterations)
            yList.append(tot_reward)
            zList.append(maximum_reward)
            makeFig()
            plt.draw()
            plt.pause(0.00001)
            #COMMENTED FOR NOW
            #fig.savefig(fileNm+'.png', bbox_inches='tight')
        '''

        end = time.time()
        tot_time = tot_time + (end-start)
        print (end-start)

    #print theta
    num_traj=1
    worldmap = None
    worldmap = np.copy(orig_worldmap)
    #rewardmap = np.array([[10,1,1],[1,1,1],[1,5,5]])
    #worldmap = np.zeros((world_map_size,world_map_size))
    #worldmap[pad_size:pad_size+reward_map_size,pad_size:pad_size+reward_map_size] = rewardmap
    curr_pos = np.array([reward_map_size,reward_map_size])
    print worldmap
    Tau = generate_trajectories(num_traj,worldmap,curr_pos,theta,isPrint=False,maxPolicy=True,rand_start=False)
    #COMMENTED FOR NOW
    if(len(sys.argv)>1):
        pickle.dump([rewardmap, gamma, Eta, num_trajectories, Tau_horizon, num_iterations, theta, Tau, xList, traj_reward_list, max_reward_list, path_max_reward_list, discount_reward_list, (float(tot_time)/num_iterations)],open(sys.argv[1]+'.pkl',"w"))
    #pickle.dump([rewardmap, gamma, Eta, num_trajectories, Tau_horizon, num_iterations, theta, Tau, xList, traj_reward_list, max_reward_list, path_max_reward_list],open(fileNm+'.pkl',"w"))
    #plot_theta(theta)
    for i in range(num_traj):
        for j in range(Tau_horizon):
            print Tau[i][j]
