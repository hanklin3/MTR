# %%
import pickle
import torch
import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from mtr.datasets.waymo.waymo_dataset import WaymoDataset
from mtr.config import cfg, cfg_from_list, cfg_from_yaml_file, log_config_to_file
from mtr.utils import common_utils

# %%
intention_points_file = 'data/waymo/cluster_64_center_dict.pkl'
cfg_file = 'tools/cfgs/waymo/mtr+20_percent_data.yaml'

# %%
intention_points = intention_query = intention_query_mlps = None



with open(intention_points_file, 'rb') as f:
    intention_points_dict = pickle.load(f)

object_type = ['TYPE_VEHICLE', 'TYPE_PEDESTRIAN', 'TYPE_CYCLIST']

intention_points = {}
for cur_type in object_type:
    cur_intention_points = intention_points_dict[cur_type]
    print('cur_intention_points', cur_intention_points.shape)
    cur_intention_points = torch.from_numpy(cur_intention_points).float().view(-1, 2)#.cuda()
    intention_points[cur_type] = cur_intention_points

# intention_query_mlps = common_layers.build_mlps(
#     c_in=d_model, mlp_channels=[d_model, d_model], ret_before_act=True
#     )

intention_points['TYPE_VEHICLE'].shape # torch.Size([64, 2])
# %%
x = intention_points['TYPE_VEHICLE'][:, 0]
y = intention_points['TYPE_VEHICLE'][:, 1]


plt.plot(x, y, '.')
plt.grid()
# %%
#############################################################################################
cfg_from_yaml_file(cfg_file, cfg)
# %%
cfg
# %%
cfg.DATA_CONFIG

# %%
dataset_cfg=cfg.DATA_CONFIG
log_file = ('./log_np_data_analysis.txt')
logger = common_utils.create_logger(log_file, rank=cfg.LOCAL_RANK)

dataset = WaymoDataset(dataset_cfg=dataset_cfg, training=True, logger=logger)
batch_size = 1
# %%
sample = dataset[3]
scenario_id = sample['scenario_id']
obj_trajs = sample['obj_trajs']
obj_trajs_pos = sample['obj_trajs_pos']
obj_trajs_mask = sample['obj_trajs_mask']
center_objects_type = sample['center_objects_type']
obj_types = sample['obj_types']
obj_trajs_future_state = sample['obj_trajs_future_state']
obj_trajs_future_mask = sample['obj_trajs_future_mask']
map_polylines = sample['map_polylines']
map_polylines_mask = sample['map_polylines_mask']
map_polylines_center = sample['map_polylines_center']

print(sample.keys())
print('scenario_id', scenario_id)
print('obj_trajs', obj_trajs.shape)
print('obj_trajs_pos', obj_trajs_pos.shape)
print('center_objects_type', center_objects_type)
print('obj_types', obj_types.shape)
print('obj_trajs_future_state', obj_trajs_future_state.shape)
print('map_polylines', map_polylines.shape)
print('map_polylines_mask', map_polylines_mask.shape)
print('map_polylines_center', map_polylines_center.shape)

"""
# dict_keys(['scenario_id', 'obj_trajs', 'obj_trajs_mask', 'track_index_to_predict', 'obj_trajs_pos', 'obj_trajs_last_pos', 'obj_types', 'obj_ids', 'center_objects_world', 'center_objects_id', 'center_objects_type', 'obj_trajs_future_state', 'obj_trajs_future_mask', 'center_gt_trajs', 'center_gt_trajs_mask', 'center_gt_final_valid_idx', 'center_gt_trajs_src', 'map_polylines', 'map_polylines_mask', 'map_polylines_center'])
scenario_id ['7e2f727866c69ea0' '7e2f727866c69ea0' '7e2f727866c69ea0'
 '7e2f727866c69ea0']
obj_trajs (4, 31, 11, 29)
obj_trajs_pos (4, 31, 11, 3)
center_objects_type ['TYPE_VEHICLE' 'TYPE_VEHICLE' 'TYPE_VEHICLE' 'TYPE_VEHICLE']
obj_types (31,)
obj_trajs_future_state (4, 31, 80, 4) # (b, cars, t, [x, y, vx, vy])
map_polylines (4, 768, 20, 9) # (num_center_objects, num_polylines, num_points_each_polyline, 9): [x, y, z, dir_x, dir_y, dir_z, global_type, pre_x, pre_y]
map_polylines_mask (4, 768, 20)
map_polylines_center (4, 768, 3)
"""
# %%
dataloader = DataLoader(
    dataset, batch_size=batch_size, shuffle=False)

# %%
def compute_angle(x, y):
    """
    Compute the angle between the first and last points of a trajectory.

    Parameters:
        x (list or numpy array): x-coordinates of the trajectory.
        y (list or numpy array): y-coordinates of the trajectory.

    Returns:
        float: Angle in degrees between the first and last points with respect to the horizontal axis.

    # counterclockwise 
    """
    if len(x) < 2 or len(y) < 2:
        raise ValueError("The trajectory must contain at least two points.")

    # Calculate start-to-end vector
    dx = x[-1] - x[-2]
    dy = y[-1] - y[-2]

    # Calculate initial and final angles
    initial_heading = np.arctan2(y[1] - y[0], x[1] - x[0])
    final_heading = np.arctan2(dy, dx)

    # # Calculate start-to-end vector
    # dx = x[-1] - x[0]
    # dy = y[-1] - y[0]

    # # Calculate initial and final angles
    # initial_heading = np.arctan2(y[1] - y[0], x[1] - x[0])
    # final_heading = np.arctan2(dy, dx)

    # Compute the difference in headings
    angle_degrees = np.degrees(final_heading - initial_heading)

    # Normalize the angle to the range [-180, 180]
    angle_degrees = (angle_degrees + 180) % 360 - 180

    return angle_degrees

def get_colormap(num_agents):
  """Compute a color map array of shape [num_agents, 4]."""
  colors = cm.get_cmap('jet', num_agents)
  colors = colors(range(num_agents))
  np.random.shuffle(colors)
  return colors


# %%
traj_past = obj_trajs_pos
traj_future = obj_trajs_future_state
num_agents = traj_past.shape[1]

for i_batch in range(obj_trajs_pos.shape[0]):
    plt.figure()
    color_map = get_colormap(num_agents)

    # past position 1s
    for i_car in range(obj_trajs_pos.shape[1]):
        # i_car = 24
        xx = obj_trajs_pos[i_batch, i_car, :, 0]
        yy = obj_trajs_pos[i_batch, i_car, :, 1]
        mask = obj_trajs_mask[i_batch, i_car, :]  # Mask (0 = invalid, 1 = valid)
 

        # Filter invalid points
        valid_indices = mask == 1
        xx = xx[valid_indices]
        yy = yy[valid_indices]
        if len(xx) > 0:
            plt.plot(xx, yy)
            # plt.plot(xx[0], yy[0], 'y^')
            # print(compute_angle(xx, yy), 'degrees')

    # future position 8s
    for i_car in range(obj_trajs_future_state.shape[1]):
        # i_car = 24
        xx = obj_trajs_future_state[i_batch, i_car, :, 0]
        yy = obj_trajs_future_state[i_batch, i_car, :, 1]
        mask = obj_trajs_future_mask[i_batch, i_car, :]  # Mask (0 = invalid, 1 = valid)
        print(color_map.shape) # (31, 4)
        print(mask.shape) # (80,)
        print(obj_trajs_future_state.shape) # (4, 31, 80, 4)
        print(obj_trajs_future_mask.shape) # (4, 31, 80)

        colors = color_map[i_car]
       
        # Filter invalid points
        valid_indices = mask == 1
        xx = xx[valid_indices]
        yy = yy[valid_indices]
        
        if len(xx) > 0:
            plt.plot(xx, yy)
       
            # plt.plot(xx[0], yy[0], 'yo')
            plt.scatter(xx[0], yy[0], s=5, marker='o', color=colors)  # `s` sets dot size
            print(compute_angle(xx, yy), 'degrees')

    for i_polyline in range(map_polylines.shape[1]):
        # Extract x, y coordinates and mask
        xx = map_polylines[i_batch, i_polyline, :, 0]  # x-coordinates
        yy = map_polylines[i_batch, i_polyline, :, 1]  # y-coordinates
        mask = map_polylines_mask[i_batch, i_polyline, :]  # Mask (0 = invalid, 1 = valid)

        # Filter valid points using the mask
        valid_indices = mask == 1
        xx_valid = xx[valid_indices]
        yy_valid = yy[valid_indices]

        plt.scatter(xx_valid[::5], yy_valid[::5], s=1, marker='.', color='blue')  # `s` sets dot size

    # map_polylines_center (4, 768, 3)
    # map_polylines_mask (4, 768, 20)
    xx = map_polylines_center[i_batch, :, 0]  # x-coordinates
    yy = map_polylines_center[i_batch, :, 1]  # y-coordinates

    print('xx', xx.shape)

    plt.scatter(xx[::], yy[::], s=1, marker='.', color='green')  # `s` sets dot size

    plt.axis('equal')
    plt.grid()

 



# %%
obj_trajs_future_mask[0]
# %%
