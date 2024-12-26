# %%
############## Package installation
# !pip install waymo-open-dataset-tf-2-12-0==1.6.4

# %%
# Imports
import os
import tarfile
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import tqdm
from waymo_open_dataset.protos import scenario_pb2
import glob

# Set matplotlib to jshtml so animations work with colab.
from matplotlib import rc
rc('animation', html='jshtml')

# %%
################## Loading the data
DATASET_FOLDER = '/home/gridsan/thlin/cameraculture_shared/hanklin/data/waymo_open_dataset_motion_v_1_2_1/uncompressed/scenario/validation/'

VALIDATION_FILES = os.path.join(DATASET_FOLDER, 'uncompressed_scenario_validation_validation.tfrecord-00000-of-00150')
VALIDATION_FILES = glob.glob(DATASET_FOLDER + '/*')
print(VALIDATION_FILES)
# %%
# Define the dataset from the TFRecords.
filenames = VALIDATION_FILES[-1]
dataset = tf.data.TFRecordDataset(filenames, compression_type='', num_parallel_reads=3)

# %% 
###############Visualization
from matplotlib import animation
from matplotlib import patches
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from waymo_open_dataset.protos import map_pb2
from waymo_open_dataset.protos import scenario_pb2

# ==== CONSTANTS ====
_ROAD_EDGE_COLOR = 'k--'
_ROAD_EDGE_ALPHA = 1.0
_ROAD_LINE_COLOR = 'k--'
_ROAD_LINE_ALPHA = 0.5
# ===================

def add_map(axis: plt.Axes,
            scenario: scenario_pb2.Scenario) -> None:
  """Adds the supported map features to a pyplot axis."""
  edges = 0
  lines = 0
  for map_feature in scenario.map_features:
    if map_feature.WhichOneof('feature_data') == 'road_edge':
      add_road_edge(axis, map_feature.road_edge)
      edges += 1
    elif map_feature.WhichOneof('feature_data') == 'road_line':
      add_road_line(axis, map_feature.road_line)
      lines += 1
    else:
      # Skip other features.
      pass
  print(f'Added {edges} road edges and {lines} road lines.')


def add_road_edge(axis: plt.Axes, road_edge: map_pb2.RoadEdge) -> None:
  """Adds a road edge to a pyplot axis."""
  x, y = zip(*[(p.x, p.y) for p in road_edge.polyline])
  axis.plot(x, y, _ROAD_EDGE_COLOR, alpha=_ROAD_EDGE_ALPHA)


def add_road_line(axis: plt.Axes, road_line: map_pb2.RoadLine) -> None:
  """Adds a road line to a pyplot axis."""
  x, y = zip(*[(p.x, p.y) for p in road_line.polyline])
  axis.plot(x, y, _ROAD_LINE_COLOR, alpha=_ROAD_LINE_ALPHA)

# %%
for cnt, data in enumerate(dataset):
  if cnt == 1:
    print(cnt)
    scenario = scenario_pb2.Scenario()
    scenario.ParseFromString(bytearray(data.numpy()))

    print('scenario_id', scenario.scenario_id)

    # Visualize scenario.
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))

    add_map(ax, scenario)

    def plot_track_trajectory(track: scenario_pb2.Track) -> None:
      #print(track)
      valids = np.array([state.valid for state in track.states])
      if np.any(valids):
        x = np.array([state.center_x for state in track.states])
        y = np.array([state.center_y for state in track.states])
        ax.plot(x[valids], y[valids], linewidth=5)

    for i, track in enumerate(scenario.tracks):
      plot_track_trajectory(track)

    break

plt.show()
# %%
