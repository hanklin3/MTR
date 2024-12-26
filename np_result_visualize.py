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
# Path to the .pkl file
# pkl_file_path = "output/waymo/mtr+100_percent_data/model1/eval/eval_with_train/epoch_30/result.pkl"
pkl_file_path = "output/waymo/mtr+100_percent_data/model1/eval/epoch_1/default/result.pkl"
pkl_file_path = "output/waymo/mtr+100_percent_data/0002_upweight2.0_time40/eval/epoch_40/default/result.pkl"

# Load the data
with open(pkl_file_path, "rb") as file:
    ordered_results = pickle.load(file)

# Print the results to verify
print(ordered_results)

# %%
print(len(ordered_results))  # 863 i_batch
print(len(ordered_results[0]))  # 1 i_objs
# %%
one_result = ordered_results[0][0]
one_result.keys()
# ['scenario_id', 'pred_trajs', 'pred_scores', 'object_id', 'object_type', 'gt_trajs', 'track_index_to_predict']
# %%
print("['scenario_id']", one_result['scenario_id'])
one_result['pred_trajs'].shape # (6, 80, 2)
one_result['pred_scores'].shape  # (6,)
one_result['gt_trajs'].shape  # (91, 10) (num_timestamps, 10)
# 10: [cx, cy, cz, dx, dy, dz, heading, vel_x, vel_y, valid]
print("['object_id']", one_result['object_id'])
print("['object_type']", one_result['object_type'])
print('[track_index_to_predict]', one_result['track_index_to_predict'])
# %%
"""
        Args:
            batch_dict:
                pred_scores: (num_center_objects, num_modes)
                pred_trajs: (num_center_objects, num_modes, num_timestamps, 7)

              input_dict:
                center_objects_world: (num_center_objects, 10)
                center_objects_type: (num_center_objects)
                center_objects_id: (num_center_objects)
                center_gt_trajs_src: (num_center_objects, num_timestamps, 10)

    single_pred_dict = {
        'scenario_id': input_dict['scenario_id'][obj_idx],
        'pred_trajs': pred_trajs_world[obj_idx, :, :, 0:2].cpu().numpy(),
        'pred_scores': pred_scores[obj_idx, :].cpu().numpy(),
        'object_id': input_dict['center_objects_id'][obj_idx],
        'object_type': input_dict['center_objects_type'][obj_idx],
        'gt_trajs': input_dict['center_gt_trajs_src'][obj_idx].cpu().numpy(),
        'track_index_to_predict': input_dict['track_index_to_predict'][obj_idx].cpu().numpy()
    }
    cur_scene_pred_list.append(single_pred_dict)
"""
# %%
def get_colormap(num_agents):
  """Compute a color map array of shape [num_agents, 4]."""
  colors = cm.get_cmap('jet', num_agents)
  colors = colors(range(num_agents))
  np.random.shuffle(colors)
  return colors

# %%
for i_batch in range(len(ordered_results)):
    plt.figure()

    if i_batch == 10:
        break

    num_agents = len(ordered_results[i_batch])
    color_map = get_colormap(num_agents)
    
    for i_obj in range(len(ordered_results[i_batch])):
        one_result = ordered_results[i_batch][i_obj]

        scenario_id = one_result['scenario_id'] 
        pred_trajs = one_result['pred_trajs'] # (6, 80, 2)
        pred_scores = one_result['pred_scores'] # (6,)
        gt_trajs = one_result['gt_trajs'] # (91, 10) (num_timestamps, 10)
        """
        one_result['pred_trajs'].shape 
        one_result['pred_scores'].shape  
        one_result['gt_trajs'].shape  
        """

        ########## past position 1s
        for i_traj in range(pred_trajs.shape[0]):
            # i_car = 24
            xx = pred_trajs[i_traj, :, 0]
            yy = pred_trajs[i_traj, :, 1]
            color = color_map[i_obj]
            score = pred_scores[i_traj]

            # if score < .1:
            #     continue

            if len(xx) > 0:

                plt.plot(xx, yy, color=color)
                # plt.plot(xx[0], yy[0], 'y^')
                # print(compute_angle(xx, yy), 'degrees')

            plt.text(xx[-1], yy[-1], f"{score:.2f}", fontsize=7, color=color, ha='center', va='bottom')  # End point


        ######### plot gt trajectory
        xx = gt_trajs[:, 0]
        yy = gt_trajs[:, 1]
        # xx = np.trim_zeros(xx, 'b')
        # yy = np.trim_zeros(yy, 'b')

        # print('xx==0', xx==0)
        # print('yy==0', yy==0)
        xx = xx[xx!=0]
        yy = yy[yy!=0]
        plt.plot(xx, yy, 'k--', markersize=5)

        plt.plot(pred_trajs[0, 0, 0], pred_trajs[0, 0, 1], marker='^', color='green')
        # plt.scatter(pred_trajs[0, 0, 0], pred_trajs[0, 0, 1], s=5, marker='^', color='yellow')

        # # future position 8s
        # for i_car in range(obj_trajs_future_state.shape[1]):
        #     # i_car = 24
        #     xx = obj_trajs_future_state[i_batch, i_car, :, 0]
        #     yy = obj_trajs_future_state[i_batch, i_car, :, 1]
        #     mask = obj_trajs_future_mask[i_batch, i_car, :]  # Mask (0 = invalid, 1 = valid)
        #     print(color_map.shape) # (31, 4)
        #     print(mask.shape) # (80,)
        #     print(obj_trajs_future_state.shape) # (4, 31, 80, 4)
        #     print(obj_trajs_future_mask.shape) # (4, 31, 80)

        #     colors = color_map[i_car]
        
        #     # Filter invalid points
        #     valid_indices = mask == 1
        #     xx = xx[valid_indices]
        #     yy = yy[valid_indices]
            
        #     if len(xx) > 0:
        #         plt.plot(xx, yy)
        
        #         # plt.plot(xx[0], yy[0], 'yo')
        #         plt.scatter(xx[0], yy[0], s=5, marker='o', color=colors)  # `s` sets dot size
        #         print(compute_angle(xx, yy), 'degrees')

    plt.axis('equal')
    plt.grid()

# %%
