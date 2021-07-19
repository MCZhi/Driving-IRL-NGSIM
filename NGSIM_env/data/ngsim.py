import os
import numpy as np
import pandas as pd
import datetime
import pytz
#from shapely.geometry import Polygon, LineString, MultiLineString, Point
#from shapely.ops import cascaded_union
# from sortedcontainers import SortedDict
from scipy.stats import hmean
from NGSIM_env.data.paras import *

GLB_DEBUG = False
GLB_ROUNDING_100MS = -2
GLB_UNIXTIME_GAP = 100
GLB_TIME_THRES = 10000
GLB_DETECT_TOL = 0.9

class ngsim_data():
  def __init__(self, name):
    self.name = name
    self.vr_dict = dict()
    self.snap_dict = dict()
    self.veh_dict = dict()
    self.snap_ordered_list = list()
    self.veh_ordered_list = list()

  def read_from_csv(self, filename):
    # format: Vehicle_ID Frame_ID Total_Frames Global_Time Local_X Local_Y Global_X Global_Y v_length v_Width v_Class v_Vel v_Acc Lane_ID	
    # O_Zone D_Zone Int_ID Section_ID Direction Movement Preceding Following Space_Headway Time_Headway Location
    f = open(filename, 'r')
    line = f.readline()
    print('Processing raw data...')
    counter = 0
    self.vr_dict = dict()
    self.snap_dict = dict()
    self.veh_dict = dict()

    while(line):
      if counter % 10000 == 0:
        print(counter)
        print(line)
      if counter > 10000 and GLB_DEBUG:
        break
      line = f.readline().strip('\n').strip('\r').strip('\t')
      if line == "":
        continue
      
      words = line.split(',')
      assert (len(words) == NUM_COLS)

      if words[GLB_loc_colidx] == self.name:
        tmp_vr = vehicle_record()
        tmp_vr.build_from_raw(counter, line)
        self.vr_dict[tmp_vr.ID] = tmp_vr
        counter += 1

        if tmp_vr.unixtime not in self.snap_dict.keys():
          self.snap_dict[tmp_vr.unixtime] = snapshot(tmp_vr.unixtime)
        self.snap_dict[tmp_vr.unixtime].add_vr(tmp_vr)

        if tmp_vr.veh_ID not in self.veh_dict.keys():
          self.veh_dict[tmp_vr.veh_ID] = vehicle(tmp_vr.veh_ID)
        self.veh_dict[tmp_vr.veh_ID].add_vr(tmp_vr)

    self.snap_ordered_list = list(self.snap_dict.keys())
    self.veh_ordered_list = list(self.veh_dict.keys())
    self.snap_ordered_list.sort()
    self.veh_ordered_list.sort()

    for tmp_unixtime, tmp_snap in self.snap_dict.items():
      tmp_snap.sort_vehs()

    for tmp_vehID, tmp_veh in self.veh_dict.items():
      tmp_veh.sort_time()

    f.close()

  def dump(self, folder, vr_filename = 'vehicle_record_file.csv', v_filename = 'vehicle_file.csv', snapshot_filename = 'snapshot_file.csv'):
    print('Dumping processed data...')
    f_vr = open(os.path.join(folder, vr_filename), 'w')
    for vr_ID, vr in self.vr_dict.items():
      f_vr.write(vr.to_string() + '\n')
    f_vr.close()

    f_v = open(os.path.join(folder, v_filename), 'w')
    for _, v in self.veh_dict.items():
      f_v.write(v.to_string() + '\n')
    f_v.close()

    f_ss = open(os.path.join(folder, snapshot_filename), 'w')
    for _, ss in self.snap_dict.items():
      f_ss.write(ss.to_string() + '\n')
    f_ss.close() 

  def load(self, folder, vr_filename = 'vehicle_record_file.csv', v_filename = 'vehicle_file.csv', snapshot_filename = 'snapshot_file.csv'):
    self.vr_dict = dict()
    self.snap_dict = dict()
    self.veh_dict = dict()
    print("Loading Data...")

    # records
    f_vr = open(os.path.join(folder, vr_filename), 'r')
    for line in f_vr:
      if line == '':
        continue
      words = line.rstrip('\n').rstrip('\r').split(',')
      assert(len(words) == 17)
      tmp_vr = vehicle_record()
      tmp_vr.build_from_processed(self.name, words)
      self.vr_dict[tmp_vr.ID] = tmp_vr
    f_vr.close()

    # vehicle
    f_v = open(os.path.join(folder, v_filename), 'r')
    for line in f_v:
      if line == '':
        continue
      words = line.rstrip('\n').rstrip('\r').split(',')
      assert(len(words) > 1)
      tmp_v = vehicle()
      tmp_v.build_from_processed(words, self.vr_dict)
      self.veh_dict[tmp_v.veh_ID] = tmp_v
    f_v.close()

    # snapshot
    f_ss = open(os.path.join(folder, snapshot_filename), 'r')
    for line in f_ss:
      if line == '':
        continue
      words = line.rstrip('\n').rstrip('\r').split(',')
      assert(len(words) > 1)
      tmp_ss = snapshot()
      tmp_ss.build_from_processed(words, self.vr_dict)
      self.snap_dict[tmp_ss.unixtime] = tmp_ss
    f_ss.close()

    # ordered list
    self.snap_ordered_list = list(self.snap_dict.keys())
    self.veh_ordered_list = list(self.veh_dict.keys())
    self.snap_ordered_list.sort()
    self.veh_ordered_list.sort()

    for tmp_unixtime, tmp_snap in self.snap_dict.items():
      tmp_snap.sort_vehs()
    for tmp_vehID, tmp_veh in self.veh_dict.items():
      tmp_veh.sort_time()

  # Especially used for us-101, clean duplicate record
  def clean(self):
    for unixtime, snap in self.snap_dict.items():
      veh_ID_list = list(map(lambda x: x.veh_ID, snap.vr_list)) 
      veh_ID_set = set(veh_ID_list)

      if len(veh_ID_list) > len(veh_ID_set):
        new_vr_list = list()
        new_vr_ID_set = set()
        for vr in snap.vr_list:
          if vr.veh_ID not in new_vr_ID_set:
            new_vr_list.append(vr)
            new_vr_ID_set.add(vr.veh_ID)
          self.snap_dict[unixtime].vr_list = new_vr_list

  def down_sample(self, sample_rate = 3000):
    self.vr_dict = {k:v for (k,v) in self.vr_dict.items() if v.unixtime % sample_rate == 0}
    self.snap_dict = {k:v for (k,v) in self.snap_dict.items() if k % sample_rate == 0}
    for veh in self.veh_dict.values():
      veh.down_sample(sample_rate)
    self.snap_ordered_list = list(filter(lambda x: x % sample_rate == 0, self.snap_ordered_list))

class vehicle_record():
  def __init__(self):
    self.ID = None
    self.veh_ID = None
    self.frame_ID = None
    self.unixtime = None

  def build_from_raw(self, ID, s1):
    self.ID = ID
    words = s1.split(',')
    assert(len(words) == NUM_COLS)

    tz = pytz.timezone(timezone_dict[words[GLB_loc_colidx]])
    self.veh_ID = np.int(words[GLB_vehID_colidx])
    #self.frame_ID = np.int(words[GLB_frmID_colidx])
    self.unixtime = np.int(words[GLB_glbtime_colidx]) 
    self.time = datetime.datetime.fromtimestamp(np.float(self.unixtime) / 1000, tz)
    self.x = np.float(words[GLB_locx_colidx])
    self.y = np.float(words[GLB_locy_colidx])
    self.lat = np.float(words[GLB_glbx_colidx])
    self.lon = np.float(words[GLB_glby_colidx])
    self.len = np.float(words[GLB_vehlen_colidx])
    self.wid = np.float(words[GLB_vehwid_colidx])
    self.cls = np.int(words[GLB_vehcls_colidx])
    self.spd = np.float(words[GLB_vehspd_colidx])
    self.acc = np.float(words[GLB_vehacc_colidx])
    self.lane_ID = np.int(words[GLB_laneID_colidx])
    #self.intersection_ID = np.int(words[GLB_interID_colidx])
    self.pred_veh_ID = np.int(words[GLB_pred_colidx])
    self.follow_veh_ID = np.int(words[GLB_follow_colidx])
    self.shead = np.float(words[GLB_shead_colidx])
    self.thead = np.float(words[GLB_thead_colidx])

  def build_from_processed(self, name, words):
    assert(len(words) == 17)
    self.ID = np.int(words[0])
    self.veh_ID = np.int(words[1])
    self.unixtime = np.int(words[2])
    tz = pytz.timezone(timezone_dict[name])
    self.time = datetime.datetime.fromtimestamp(np.float(self.unixtime) / 1000, tz)
    self.x = np.float(words[3])
    self.y = np.float(words[4])
    self.lat = np.float(words[5])
    self.lon = np.float(words[6])
    self.len = np.float(words[7])
    self.wid = np.float(words[8])
    self.cls = np.int(words[9])
    self.spd = np.float(words[10])
    self.acc = np.float(words[11])
    self.lane_ID = np.int(words[12])
    self.pred_veh_ID = np.int(words[13])
    self.follow_veh_ID = np.int(words[14])
    self.shead = np.float(words[15])
    self.thead = np.float(words[16])

  def __str__(self):
    return ("Vehicle record: {}, vehicle ID: {}, unixtime: {}, time: {}, lane: {}, y: {}, x: {}".format(self.ID, self.veh_ID, self.unixtime, 
              self.time.strftime("%Y-%m-%d %H:%M:%S"), self.lane_ID, self.y, self.x))

  def __repr__(self):
    return self.__str__()

  def to_string(self):
    return ','.join([str(e) for e in [self.ID, self.veh_ID, self.unixtime, 
                                      self.x, self.y, self.lat, self.lon,
                                      self.len, self.wid, self.cls,
                                      self.spd, self.acc, self.lane_ID,
                                      self.pred_veh_ID, self.follow_veh_ID, self.shead, self.thead]])

class snapshot():
  def __init__(self, unixtime = None):
    self.unixtime = unixtime
    self.vr_list = list()

  def build_from_processed(self, words, vr_dict):
    assert(len(words) > 1)
    self.unixtime = np.int(words[0])
    self.vr_list = list(map(lambda x: vr_dict[np.int(x)], words[1:]))

  def add_vr(self, vr):
    assert (vr.unixtime == self.unixtime)
    self.vr_list.append(vr)

  def sort_vehs(self, ascending = True):
    self.vr_list = sorted(self.vr_list, key = lambda x: (x.y, x.lane_ID), reverse = (not ascending))

  def __str__(self):
    return ("Snapshot: unixtime: {}, number of vehs: {}".format(self.unixtime, len(self.vr_list)))
  
  def __repr__(self):
    return self.__str__()

  def to_string(self):
    return ','.join([str(e) for e in [self.unixtime] + list(map(lambda x: x.ID, self.vr_list))])

class vehicle():
  def __init__(self, veh_ID = None):
    self.veh_ID = veh_ID
    self.vr_list = list()
    self.trajectory = list()

  def build_from_processed(self, words, vr_dict):
    assert(len(words) > 1)
    self.veh_ID = np.int(words[0])
    self.vr_list = list(map(lambda x: vr_dict[np.int(x)], words[1:]))

  def add_vr(self, vr):
    assert (vr.veh_ID == self.veh_ID)
    self.vr_list.append(vr)

  def sort_time(self, ascending = True):
    self.vr_list = sorted(self.vr_list, key = lambda x: (x.unixtime), reverse = (not ascending))

  def __str__(self):
    return ("Vehicle: veh_ID: {}, number of unixtimes: {}".format(self.veh_ID, len(self.vr_list)))
  
  def __repr__(self):
    return self.__str__()

  def to_string(self):
    return ','.join([str(e) for e in [self.veh_ID] + list(map(lambda x: x.ID, self.vr_list))])

  # downsampl, interval unit: ms
  def down_sample(self, sample_rate): 
    # self.sampled_vr_list = list()
    # cur_time = (np.round(np.random.rand() * interval + GLB_UNIXTIME_GAP/2, GLB_ROUNDING_100MS) 
    #                       + self.vr_list[0].unixtime)
    # for tmp_vr in self.vr_list():
      # if tmp_vr.unixtime - cur_time >= interval:
        # self.sampled_vr_list.append(tmp_vr)
        # cur_time = tmp_vr.unixtime
    self.vr_list = list(filter(lambda x: x.unixtime % sample_rate == 0, self.vr_list))

  def get_stayed_lanes(self):
    return list(set(list(map(lambda x: x.lane_ID, self.vr_list))))

  #def _get_lane_separated_vrs(self, name):
  #  lane2vr_dict = dict()
  #  # stayed_lanes = self._get_stayed_lanes()
  #  for vr in self.vr_list:
  #    if vr.lane_ID in GLB_LANE_CONSIDERED[name]:
  #      if vr.lane_ID not in lane2vr_dict.keys():
  #        lane2vr_dict[vr.lane_ID] = list()
  #      lane2vr_dict[vr.lane_ID].append(vr)
  #  return lane2vr_dict

  def build_trajectory(self):
    #lane2vr_dict = self._get_lane_separated_vrs(name)

    #for lane_ID, tmp_vr_list in lane2vr_dict.items():
      #print (lane_ID)
    #tmp_traj = trajectory(GLB_TIME_THRES)
    #tmp_traj.construct_trajectory(self.vr_list)
      #print (self.vr_list)
      #print (tmp_traj.trajectory_list)
    #tmp_traj.build_poly_list()
    #self.trajectory = tmp_traj
        
    vr_list = self.vr_list
    assert (len(vr_list) > 0)
    self.trajectory = list()
    cur_time = vr_list[0].unixtime
    tmp_trj = [vr_list[0]]

    for tmp_vr in vr_list[1:]:
      if tmp_vr.unixtime - cur_time > GLB_TIME_THRES:
        if len(tmp_trj) > 1:
          self.trajectory.append(tmp_trj)
        tmp_trj = [tmp_vr]
      else:
        tmp_trj.append(tmp_vr)
      cur_time = tmp_vr.unixtime
      
    if len(tmp_trj) > 1:
      self.trajectory.append(tmp_trj)

class trajectory():
  def __init__(self, thres):
    self.threshold = thres
    self.trajectory_list = list()
    self.polygon_list = list()
    self.polyline_list = list()

  def construct_trajectory(self, vr_list):
    # print (vr_list)
    assert (len(vr_list) > 0)
    self.trajectory_list = list()
    cur_time = vr_list[0].unixtime
    tmp_trj = [vr_list[0]]
    for tmp_vr in vr_list[1:]:
      if tmp_vr.unixtime - cur_time > self.threshold:
        if len(tmp_trj) > 1:
          self.trajectory_list.append(tmp_trj)
        tmp_trj = [tmp_vr]
      else:
        tmp_trj.append(tmp_vr)
      cur_time = tmp_vr.unixtime
    if len(tmp_trj) > 1:
      self.trajectory_list.append(tmp_trj)

  def build_poly_list(self):
    self.polygon_list = list()
    if len(self.trajectory_list) > 0:
      for traj in self.trajectory_list:
        tmp_polyline, tmp_polygon = self._build_poly(traj)
        if tmp_polygon.is_valid and tmp_polyline.is_valid:
          self.polyline_list.append(tmp_polyline)
          self.polygon_list.append(tmp_polygon)
        else:
          print ('Warnning: invalid polygon')

  def _build_poly(self, traj):
    assert(len(traj) > 1)
    point_list = list()
    for i in range(len(traj)):
      point_list.append((traj[i].unixtime, traj[i].y))
    tmp_polyline = LineString(point_list)
    for i in reversed(range(len(traj))):
      if traj[i].shead == 0: 
        point_list.append((traj[i].unixtime, traj[i].y + 1000))
      else:
        point_list.append((traj[i].unixtime, traj[i].y + traj[i].shead))
    p = Polygon(point_list)
    # print (p)
    # assert(p.is_valid)
    return tmp_polyline, p

class lidar():
  def __init__(self, veh_ID = None, r= None):
    self.veh_ID = veh_ID
    self.r = r

  def get_detected_range(self, vr):
    circle = Point(vr.y, vr.x).buffer(self.r)
    return circle

  def get_detected_vr_list(self, vr, vr_list, mis_rate):
    assert(vr.veh_ID == self.veh_ID)
    c = self.get_detected_range(vr)
    detected_vr_list = list()
    for vr in vr_list:
      p = Point(vr.y, vr.x)
      if c.intersects(p) and np.random.rand() >= mis_rate:
        detected_vr_list.append(vr)
    return detected_vr_list

class monitor_center():
  def __init__(self, min_space, max_space, min_time, max_time, miss_rate = 0.0, spd_noise = 0.0, method = 'Detecting'):
    self.lidar_dict = dict()
    self.detection_record = dict()
    self.min_space = min_space
    self.max_space = max_space
    self.min_time = min_time
    self.max_time = max_time
    self.method = method
    self.miss_rate = miss_rate
    self.spd_noise = spd_noise

  def install_lidar(self, veh_list, r_list):
    assert(len(veh_list) == len(r_list))
    self.lidar_dict = dict()
    for i in range(len(veh_list)):
      veh = veh_list[i]
      r = r_list[i]
      self.lidar_dict[veh.veh_ID] = lidar(veh.veh_ID, r)

  def detect_all_snap(self, snap_dict):
    self.detection_record = dict()
    for unixtime, snap in snap_dict.items():
      if snap.unixtime < self.min_time or snap.unixtime > self.max_time:
        continue
      # print (unixtime, snap)
      tmp_dict = self._detect_one_snap(snap, self.miss_rate)
      if len(tmp_dict) > 0:
        self.detection_record[unixtime] = tmp_dict

  def _detect_one_snap(self, snap, mis_rate):
    tmp_dict = dict()    
    for potential_lidar_vr in snap.vr_list:
      if potential_lidar_vr.veh_ID in self.lidar_dict.keys():
        detected_vr_list = self.lidar_dict[potential_lidar_vr.veh_ID].get_detected_vr_list(potential_lidar_vr, snap.vr_list, mis_rate)
        c = self.lidar_dict[potential_lidar_vr.veh_ID].get_detected_range(potential_lidar_vr)
        # print (detected_vr_list)
        if len(detected_vr_list)> 0:
          tmp_dict[potential_lidar_vr]= (c, detected_vr_list)

    if self.method == 'Detecting':
      return tmp_dict
    if self.method == 'Tracking':
      tmp_dict2 = dict()
      tmp_tot_list = list()
      tmp_c_list = list()
      for potential_lidar_vr in tmp_dict.keys():
        tmp_tot_list += tmp_dict[potential_lidar_vr][1]
        tmp_c_list.append(tmp_dict[potential_lidar_vr][0])
      union_c = cascaded_union(tmp_c_list)
      tmp_dict2[0] = (union_c, list(set(tmp_tot_list)))
      return tmp_dict2
    raise("Error, not implemented")

  def reduce_to_mesh(self, m, name):
    for unixtime in self.detection_record.keys():
      for lidar_vr in self.detection_record[unixtime].keys():
        lane2vr_dict = get_lane_separated_vr_list(self.detection_record[unixtime][lidar_vr][1], name)
        for lane_ID, tmp_vr_list in lane2vr_dict.items():
          # tot_count = len(self.detection_record[unixtime][lidar_vr])
          # tmp_spd_list = list(map(lambda x: x.spd, self.detection_record[unixtime][lidar_vr]))
          tmp_dict = dict()
          for tmp_vr in tmp_vr_list:
            if not m.is_in(lane_ID, unixtime, tmp_vr.y):
              continue
            (i,j,k) = m.locate(lane_ID, unixtime, tmp_vr.y)
            if j not in tmp_dict.keys():
              tmp_dict[j] = dict()
            if k not in tmp_dict[j].keys():
              tmp_dict[j][k] = list()
            if tmp_vr.spd > 0:
              tmp_dict[j][k].append(tmp_vr.spd + tmp_vr.spd * np.random.uniform(-1, 1) * self.spd_noise)
          for j in tmp_dict.keys():
            for k in tmp_dict[j].keys():
              if len(tmp_dict[j][k]) > 0:
                m.mesh_storage[lane_ID][j][k][2].append(len(tmp_dict[j][k]))
                m.mesh_storage[lane_ID][j][k][3].append(hmean(np.array(tmp_dict[j][k])))

  def reduce_to_mesh2(self, m, sm, name):
    for unixtime in self.detection_record.keys():
      k = None
      for lidar_vr in self.detection_record[unixtime].keys():
        lane2vr_dict = get_lane_separated_vr_list(self.detection_record[unixtime][lidar_vr][1], name)
        tmp_dict = dict()
        for lane_ID, tmp_vr_list in lane2vr_dict.items():
          # tot_count = len(self.detection_record[unixtime][lidar_vr])
          # tmp_spd_list = list(map(lambda x: x.spd, self.detection_record[unixtime][lidar_vr]))
          tmp_dict[lane_ID] = dict()
          for tmp_vr in tmp_vr_list:
            if not m.is_in(lane_ID, unixtime, tmp_vr.y):
              continue
            (i,j,k) = m.locate(lane_ID, unixtime, tmp_vr.y)
            if j not in tmp_dict[lane_ID].keys():
              tmp_dict[lane_ID][j] = dict()
            if k not in tmp_dict[lane_ID][j].keys():
              tmp_dict[lane_ID][j][k] = list()
            tmp_dict[lane_ID][j][k].append(tmp_vr)

        if k is None:
          continue

        for i in sm.mesh_storage.keys():
          for j in sm.mesh_storage[i].keys():
            tmp_l = sm.mesh_storage[i][j]
            detected_lane = tmp_l.intersection(self.detection_record[unixtime][lidar_vr][0])
            if (not detected_lane.is_empty) and detected_lane.length > 0:
              tmp_portion = np.float(detected_lane.length) / np.float(tmp_l.length)
              if tmp_portion > GLB_DETECT_TOL:
                # print (tmp_portion)
                if i in tmp_dict.keys() and j in tmp_dict[i].keys():
                    m.mesh_storage[i][j][k][2].append(np.float(len(tmp_dict[i][j][k])))
                    spd_list = list(filter(lambda x: x>0, map(lambda x: x.spd + x.spd * np.random.uniform(-1, 1) * self.spd_noise, tmp_dict[i][j][k])))
                    if len(spd_list) > 0:
                      m.mesh_storage[i][j][k][3].append(hmean(np.array(spd_list)))
                  # else:
                  #   m.mesh_storage[i][j][k][2].append(0.0)
                else:
                  m.mesh_storage[i][j][k][2].append(0.0)

class space_mesh():
  def __init__(self, num_spatial_cells = None, name = None):
    self.num_spatial_cells = num_spatial_cells
    self.name = name
    self.lane_centerline = dict()
    self.mesh_storage = dict()

  def init_mesh(self, min_space, max_space):
    assert(self.num_spatial_cells is not None)
    assert(self.name is not None)
    self.mesh_storage = dict()
    self.min_space = min_space
    self.max_space = max_space
    space_breaks = np.linspace(min_space, max_space, self.num_spatial_cells + 1)
    for i in GLB_LANE_CONSIDERED[self.name]:
      self.mesh_storage[i] = dict()
      for j in range(self.num_spatial_cells):
        l = LineString([(space_breaks[j], self.lane_centerline[i]), 
                      (space_breaks[j+1], self.lane_centerline[i])])
        self.mesh_storage[i][j] = l

  def build_lane_centerline(self, snap_dict, min_time, max_time):
    self.lane_centerline = dict()
    tmp_dict = dict()
    for snap in snap_dict.values():
      if snap.unixtime < min_time or snap.unixtime > max_time:
        continue
      for vr in snap.vr_list:
        if vr.lane_ID not in tmp_dict.keys():
          tmp_dict[vr.lane_ID] = list()
        tmp_dict[vr.lane_ID].append(vr.x)
    for lane_ID, l in tmp_dict.items():
      self.lane_centerline[lane_ID] = np.median(np.array(l))


class mesh():
  def __init__(self, num_spatial_cells = None, num_temporal_cells = None, name = None):
    self.num_spatial_cells = num_spatial_cells
    self.num_temporal_cells = num_temporal_cells
    self.name = name
    self.mesh_storage = dict()
    self.lane_qkv = dict()
    self.min_space = None
    self.max_space = None
    self.min_time = None
    self.max_time = None
    self.num_lane = len(GLB_LANE_CONSIDERED[self.name])

  def init_mesh(self, min_space, max_space, min_time, max_time):
    assert(self.num_spatial_cells is not None)
    assert(self.num_temporal_cells is not None)
    assert(self.name is not None)
    self.min_space = min_space
    self.max_space = max_space
    self.min_time = min_time
    self.max_time = max_time
    self.mesh_storage = dict()
    space_breaks = np.linspace(min_space, max_space, self.num_spatial_cells + 1)
    time_breaks = np.linspace(min_time, max_time, self.num_temporal_cells + 1)
    for i in GLB_LANE_CONSIDERED[self.name]:
      self.mesh_storage[i] = dict()
      for j in range(self.num_spatial_cells):
        self.mesh_storage[i][j] = dict()
        for k in range(self.num_temporal_cells):
          tmp_p = Polygon([(time_breaks[k], space_breaks[j]), (time_breaks[k+1], space_breaks[j]), 
                            (time_breaks[k+1], space_breaks[j+1]), (time_breaks[k], space_breaks[j+1])])
          #[polygon, area, time, distance, q, k, v]
          self.mesh_storage[i][j][k] = [tmp_p, [], [], [], None, None, None]

  def locate(self, lane_ID, unixtime, y):
    # print (unixtime)
    # print (self.min_time, self.max_time)
    assert(lane_ID in self.mesh_storage.keys())
    assert(unixtime >= self.min_time and unixtime <= self.max_time)
    assert(y >= self.min_space and y <= self.max_space)
    i = lane_ID
    j = np.int((y - 0.001 - self.min_space) / (np.float(self.max_space - self.min_space)/np.float(self.num_spatial_cells)))
    # print (j, y, self.min_space, self.max_space,self.num_spatial_cells)
    assert (j < self.num_spatial_cells)
    k = np.int((unixtime - 0.001 - self.min_time) / (np.float(self.max_time - self.min_time)/ np.float(self.num_temporal_cells)))
    assert (k < self.num_temporal_cells)
    return (i,j,k)

  def is_in(self, lane_ID, unixtime, y):
    if lane_ID not in self.mesh_storage.keys():
      return False
    if not (unixtime >= self.min_time and unixtime <= self.max_time):
      return False
    if not (y >= self.min_space and y <= self.max_space):
      return False
    return True

  def update_vehilce(self, v):
    for lane_ID in v.trajectory.keys():
      tmp_traj = v.trajectory[lane_ID]
      for j in self.mesh_storage[lane_ID].keys():
        for k in self.mesh_storage[lane_ID][j].keys():
          tmp_poly = self.mesh_storage[lane_ID][j][k][0]
          assert(len(tmp_traj.polygon_list) == len(tmp_traj.polyline_list))
          for i in range(len(tmp_traj.polygon_list)):
            v_poly = tmp_traj.polygon_list[i]
            v_line = tmp_traj.polyline_list[i]

            tmp_v_line = tmp_poly.intersection(v_line)
            # print (tmp_poly.exterior.coords.xy)
            # # print (v_line)
            # print (type(tmp_v_line))
            # if type(tmp_v_line) == MultiLineString:
            #   print (list(tmp_v_line.geoms))
            # print (tmp_v_line.is_empty)
            if not tmp_v_line.is_empty:
              if type(tmp_v_line) == LineString and len(tmp_v_line.coords) > 1:
                self.mesh_storage[lane_ID][j][k][2].append(tmp_v_line.coords[-1][0] - tmp_v_line.coords[0][0])
                self.mesh_storage[lane_ID][j][k][3].append(tmp_v_line.coords[-1][1] - tmp_v_line.coords[0][1])

                tmp_area = tmp_poly.intersection(v_poly).area
                assert(tmp_area>0)
                self.mesh_storage[lane_ID][j][k][1].append(tmp_area)

  def update_qkv(self):
    for i in self.mesh_storage.keys():
      for j in self.mesh_storage[i].keys():
        for k in self.mesh_storage[i][j].keys():
          if len(self.mesh_storage[i][j][k][1]) > 0:
            ave_area = np.mean(np.array(self.mesh_storage[i][j][k][1]))
            ave_time = np.mean(np.array(self.mesh_storage[i][j][k][2]))
            ave_dis = np.mean(np.array(self.mesh_storage[i][j][k][3]))
            self.mesh_storage[i][j][k][4] = ave_dis / ave_area #q, volue
            self.mesh_storage[i][j][k][5] = ave_time / ave_area #k, density
            self.mesh_storage[i][j][k][6] = ave_dis / ave_time #v, speed
          else:
            self.mesh_storage[i][j][k][4] = np.nan
            self.mesh_storage[i][j][k][5] = np.nan
            self.mesh_storage[i][j][k][6] = np.nan

    self.lane_qkv = dict()
    for i in self.mesh_storage.keys():
      self.lane_qkv[i] = list()
      self.lane_qkv[i].append(np.nan * np.ones(shape=(self.num_spatial_cells, self.num_temporal_cells)))
      self.lane_qkv[i].append(np.nan * np.ones(shape=(self.num_spatial_cells, self.num_temporal_cells)))
      self.lane_qkv[i].append(np.nan * np.ones(shape=(self.num_spatial_cells, self.num_temporal_cells)))
      for j in self.mesh_storage[i].keys():
        for k in self.mesh_storage[i][j].keys():
          self.lane_qkv[i][0][self.num_spatial_cells-1-j,k] = self.mesh_storage[i][j][k][4]
          self.lane_qkv[i][1][self.num_spatial_cells-1-j,k] = self.mesh_storage[i][j][k][5]
          self.lane_qkv[i][2][self.num_spatial_cells-1-j,k] = self.mesh_storage[i][j][k][6]

  def update_qkv2(self):
    for i in self.mesh_storage.keys():
      for j in self.mesh_storage[i].keys():
        for k in self.mesh_storage[i][j].keys():
          if len(self.mesh_storage[i][j][k][2]) and len(self.mesh_storage[i][j][k][3]) > 0:
            ave_k = (np.mean(np.array(self.mesh_storage[i][j][k][2])) 
                      / (np.float(self.max_space - self.min_space)/ np.float(self.num_spatial_cells)))
            ave_v = np.mean(np.array(self.mesh_storage[i][j][k][3])) / 1000
            self.mesh_storage[i][j][k][4] = ave_k * ave_v#q, volue
            self.mesh_storage[i][j][k][5] = ave_k #k, density
            self.mesh_storage[i][j][k][6] = ave_v #v, speed
          else:
            self.mesh_storage[i][j][k][4] = np.nan
            self.mesh_storage[i][j][k][5] = np.nan
            self.mesh_storage[i][j][k][6] = np.nan

    self.lane_qkv = dict()
    for i in self.mesh_storage.keys():
      self.lane_qkv[i] = list()
      self.lane_qkv[i].append(np.nan * np.ones(shape=(self.num_spatial_cells, self.num_temporal_cells)))
      self.lane_qkv[i].append(np.nan * np.ones(shape=(self.num_spatial_cells, self.num_temporal_cells)))
      self.lane_qkv[i].append(np.nan * np.ones(shape=(self.num_spatial_cells, self.num_temporal_cells)))
      for j in self.mesh_storage[i].keys():
        for k in self.mesh_storage[i][j].keys():
          self.lane_qkv[i][0][self.num_spatial_cells-1-j,k] = self.mesh_storage[i][j][k][4]
          self.lane_qkv[i][1][self.num_spatial_cells-1-j,k] = self.mesh_storage[i][j][k][5]
          self.lane_qkv[i][2][self.num_spatial_cells-1-j,k] = self.mesh_storage[i][j][k][6]

def get_lane_separated_vr_list(vr_list, name):
  lane2vr_dict = dict()
  # stayed_lanes = self._get_stayed_lanes()
  for vr in vr_list:
    if vr.lane_ID in GLB_LANE_CONSIDERED[name]:
      if vr.lane_ID not in lane2vr_dict.keys():
        lane2vr_dict[vr.lane_ID] = list()
      lane2vr_dict[vr.lane_ID].append(vr)
  return lane2vr_dict

def clone_part_mesh(m):
  m2 = mesh(num_spatial_cells = m.num_spatial_cells, num_temporal_cells = m.num_temporal_cells, name = m.name)
  # m2.init_mesh(m.min_space, m.max_space, m.min_time, m.max_time)
  m2.lane_qkv = dict()
  for i in m.mesh_storage.keys():
    m2.lane_qkv[i] = list()
    m2.lane_qkv[i].append(m.lane_qkv[i][0].copy())
    m2.lane_qkv[i].append(m.lane_qkv[i][1].copy())
    m2.lane_qkv[i].append(m.lane_qkv[i][2].copy())
  return m2