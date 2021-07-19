import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from NGSIM_env.data.ngsim import *

def trajectory_smoothing(trajectory):
    trajectory = np.array(trajectory)
    x = trajectory[:,0]
    y = trajectory[:,1]
    speed = trajectory[:,2]
    lane = trajectory[:,3]

    window_length = 21 if len(x[np.nonzero(x)]) >= 21 else len(x[np.nonzero(x)]) if len(x[np.nonzero(x)]) % 2 !=0 else len(x[np.nonzero(x)])-1
    x[np.nonzero(x)] = signal.savgol_filter(x[np.nonzero(x)], window_length=window_length, polyorder=3) # window size used for filtering, order of fitted polynomial
    y[np.nonzero(y)] = signal.savgol_filter(y[np.nonzero(y)], window_length=window_length, polyorder=3)
    speed[np.nonzero(speed)] = signal.savgol_filter(speed[np.nonzero(speed)], window_length=window_length, polyorder=3)
   
    return [[float(x), float(y), float(s), int(l)] for x, y, s, l in zip(x, y, speed, lane)]

def build_trajecotry(scene, period, vehicle_ID):
    ng = ngsim_data(scene)
    ng.load('NGSIM_env/data/processed/'+scene)
    records = ng.vr_dict
    vehicles = ng.veh_dict
    snapshots = ng.snap_dict
    surroundings = []
    record_trajectory = {'ego':{'length':0, 'width':0, 'trajectory':[]}}
    
    for veh_ID, v in vehicles.items():
        v.build_trajectory()

    ego_trajectories = vehicles[vehicle_ID].trajectory
    selected_trajectory = ego_trajectories[period]

    D = 50 if scene == 'us-101' else 20

    ego = []
    nearby_IDs = []
    for position in selected_trajectory:
        record_trajectory['ego']['length'] = position.len
        record_trajectory['ego']['width'] = position.wid
        ego.append([position.x, position.y, position.spd, position.lane_ID])
        records = snapshots[position.unixtime].vr_list
        other = []
        for record in records:
            if record.veh_ID != vehicle_ID:
                other.append([record.veh_ID, record.len, record.wid, record.x, record.y, record.spd, record.lane_ID])
                d = abs(position.y - record.y)
                if d <= D:            
                    nearby_IDs.append(record.veh_ID)
        surroundings.append(other)
        
    record_trajectory['ego']['trajectory'] = ego

    for v_ID in set(nearby_IDs):
        record_trajectory[v_ID] = {'length':0, 'width':0, 'trajectory':[]}
    
    # fill in data
    for timestep_record in surroundings:
        scene_IDs = []
        for vehicle_record in timestep_record:
            v_ID = vehicle_record[0]
            v_length = vehicle_record[1]
            v_width = vehicle_record[2]
            v_x = vehicle_record[3]
            v_y = vehicle_record[4]
            v_s = vehicle_record[5]
            v_laneID = vehicle_record[6]
            if v_ID in set(nearby_IDs):
                scene_IDs.append(v_ID)
                record_trajectory[v_ID]['length'] = v_length
                record_trajectory[v_ID]['width'] = v_width
                record_trajectory[v_ID]['trajectory'].append([v_x, v_y, v_s, v_laneID])
        for v_ID in set(nearby_IDs):
            if v_ID not in scene_IDs:
                record_trajectory[v_ID]['trajectory'].append([0, 0, 0, 0])
    
    # trajectory smoothing
    for key in record_trajectory.keys():
        orginal_trajectory = record_trajectory[key]['trajectory']
        smoothed_trajectory = trajectory_smoothing(orginal_trajectory)
        record_trajectory[key]['trajectory'] = smoothed_trajectory

    return record_trajectory

if __name__ == "__main__":
    scene = 'us-101'

    for id in np.random.choice(3200, size=300, replace=False):
        try:
            trajectory_set = build_trajecotry(scene, 0, id+1)
        except:
            continue

        ego = trajectory_set['ego']['trajectory']
        trajectory = np.array(ego)
        plt.plot(trajectory[:,1]/3.281, trajectory[:,0]/3.281)
    
    plt.gca().set_aspect('auto', 'datalim')
    plt.gca().set(xlim=(0, 660), ylim=(0, 25))
    plt.gca().invert_yaxis()
    plt.xlabel('Longitudinal position [m]', fontsize=20)
    plt.ylabel('Lateral position [m]', fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.tight_layout()
    plt.show()
