from NGSIM_env.envs.ngsim_env import NGSIMEnv
import matplotlib.pyplot as plt
from NGSIM_env.utils import *
import numpy as np
import csv
from NGSIM_env.utils import *

# parameters
n_iters = 200
lr = 0.05
lam = 0.01
feature_num = 8
vehicle_id = 317
period = 0
render_env = False

# create environment
print('Target Vehicle: {}'.format(vehicle_id))
env = NGSIMEnv(scene='us-101', period=period, vehicle_id=vehicle_id, IDM=False)
    
# reset environment
env.reset()
length = env.vehicle.ngsim_traj.shape[0]

# create training log
with open('training_log_{}.csv'.format(vehicle_id), 'w') as csvfile:  
    csvwriter = csv.writer(csvfile)  
    csvwriter.writerow(['iteration', 'human feature', 'trajectory feature', 'feature norm', 'weights', 'human likeness', 'mean human likeness', 'likelihood'])  

# Data collection
buffer = []
human_traj_features = []
timesteps = np.linspace(10, length-60, num=50, dtype=np.int16)
train_steps = np.random.choice(timesteps, size=35, replace=False)
test_steps = [t for t in timesteps if t not in train_steps]

# run training set
for start in train_steps:
    # go to the scene
    env.reset(reset_time=start)

    # determine target sampling space
    lateral_offsets, target_speeds = env.sampling_space()
            
    # set up buffer of the scene
    buffer_scene = []

    # lateral and speed trajectory sampling
    print('scene: {}, sampling...'.format(start))
    for lateral in lateral_offsets:
        for target_speed in target_speeds:
            # sample a trajectory
            action = (lateral, target_speed, 5)
            obs, features, terminated, info = env.step(action)

            # render env
            if render_env:
                env.render()
                    
            # get the features
            traj_features = features[:-1]
            human_likeness = features[-1]

            # add the trajectory to scene buffer
            buffer_scene.append([lateral, target_speed, traj_features, human_likeness])

            # go back to original scene
            env.reset(reset_time=start)
        
    # calculate human trajectory feature
    env.reset(reset_time=start, human=True)
    obs, features, terminated, info = env.step()

    # eliminate invalid examples
    if terminated or features[-1] > 2.5:
        continue

    # process data
    human_traj = features[:-1]
    buffer_scene.append([0, 0, features[:-1], features[-1]])

    # add generated and human trajectories to buffer
    human_traj_features.append(human_traj)
    buffer.append(buffer_scene)

# normalize features
max_v = np.max([traj[2] for traj in buffer_scene for buffer_scene in buffer], axis=0)
min_v = np.min([traj[2] for traj in buffer_scene for buffer_scene in buffer], axis=0)
max_v[6] = 1.0

for scene in buffer:
    for traj in scene:
        for i in range(feature_num):
            traj[2][i] /= max_v[i]

#### MaxEnt IRL ####
# initialize weights
theta = np.random.normal(0, 0.05, size=feature_num)

# iterations
beta1 = 0.9; beta2 = 0.999; eps = 1e-8
pm = None
pv = None
grad_log = []
human_likeness_log = []

for iteration in range(n_iters):
    print('iteration: {}/{}'.format(iteration+1, n_iters))

    # fix collision feature's weight
    theta[-2] = -10

    feature_exp = np.zeros([feature_num])
    human_feature_exp = np.zeros([feature_num])
    index = 0
    log_like_list = []
    iteration_human_likeness = []
    num_traj = 0

    for scene in buffer:
        # compute on each scene
        scene_trajs = []
        for trajectory in scene:
            reward = np.dot(trajectory[2], theta)
            scene_trajs.append((reward, trajectory[2], trajectory[3])) # reward, feature vector, human likeness

        # calculate probability of each trajectory
        rewards = [traj[0] for traj in scene_trajs]
        probs = [np.exp(reward) for reward in rewards]
        probs = probs / np.sum(probs)

        # calculate feature expectation with respect to the weights
        traj_features = np.array([traj[1] for traj in scene_trajs])
        feature_exp += np.dot(probs, traj_features) # feature expectation

        # calculate likelihood
        log_like = np.log(probs[-1] / np.sum(probs))
        log_like_list.append(log_like)

        # select trajectories tp calculate human likeness
        idx = probs.argsort()[-3:][::-1]
        iteration_human_likeness.append(np.min([scene_trajs[i][-1] for i in idx]))
            
        # calculate human trajectory feature
        human_feature_exp += human_traj_features[index]

        # go to next trajectory
        num_traj += 1
        index += 1
        
    # compute gradient
    grad = human_feature_exp - feature_exp - 2*lam*theta
    grad = np.array(grad, dtype=np.float)

    # update weights
    if pm is None:
        pm = np.zeros_like(grad)
        pv = np.zeros_like(grad)

    pm = beta1 * pm + (1 - beta1) * grad
    pv = beta2 * pv + (1 - beta2) * (grad*grad)
    mhat = pm / (1 - beta1**(iteration+1))
    vhat = pv / (1 - beta2**(iteration+1))
    update_vec = mhat / (np.sqrt(vhat) + eps)
    theta += lr * update_vec

    # add to training log
    with open('training_log_{}.csv'.format(vehicle_id), 'a') as csvfile:  
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow([iteration+1, np.array(human_feature_exp/num_traj), np.array(feature_exp/num_traj), np.linalg.norm(human_feature_exp/num_traj - feature_exp/num_traj),
                            theta, iteration_human_likeness, np.mean(iteration_human_likeness), np.sum(log_like_list)/num_traj])

# run test
# create testing log
with open('testing_log_{}.csv'.format(vehicle_id), 'w') as csvfile:  
    csvwriter = csv.writer(csvfile)  
    csvwriter.writerow(['scene', 'human likeness', 'weights', 'max features', 'min features', 'FDE']) 

# begin testing
for start in test_steps:
    # go to the scene
    env.reset(reset_time=start)

    # determine target sampling space
    lateral_offsets, target_speeds = env.sampling_space()
            
    # set up buffer of the scene
    buffer_scene = []

    # lateral and speed trajectory sampling
    print('scene: {}, sampling...'.format(start))
    for lateral in lateral_offsets:
        for target_speed in target_speeds:
            # sample a trajectory
            action = (lateral, target_speed, 5)
            obs, features, terminated, info = env.step(action)

            # render env
            if render_env:
                env.render()
                    
            # get the features
            traj_features = features[:-1]
            human_likeness = features[-1]

            # add the trajectory to scene buffer
            buffer_scene.append([lateral, target_speed, traj_features, human_likeness])

            # go back to original scene
            env.reset(reset_time=start)

    # normalize features 
    for traj in buffer_scene:
        for i in range(feature_num):
            if max_v[i] == 0:
                traj[2][i] = 0
            else:
                traj[2][i] /= max_v[i]
        
    # evaluate trajectories
    reward_HL = []
    for trajectory in buffer_scene:
        reward = np.dot(trajectory[2], theta)
        reward_HL.append([reward, trajectory[3]]) # reward, human likeness

    # calculate probability of each trajectory
    rewards = [traj[0] for traj in reward_HL]
    probs = [np.exp(reward) for reward in rewards]
    probs = probs / np.sum(probs)

    # select trajectories to calculate human likeness
    idx = probs.argsort()[-3:][::-1]
    HL = np.min([reward_HL[i][-1] for i in idx])

    # add to testing log
    with open('testing_log_{}.csv'.format(vehicle_id), 'a') as csvfile:   
        csvwriter = csv.writer(csvfile)  
        csvwriter.writerow([start, HL, theta, max_v, min_v, [reward_HL[i][-1] for i in range(len(reward_HL))]])