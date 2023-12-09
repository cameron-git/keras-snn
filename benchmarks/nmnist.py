import tonic
from . import utils

BATCH_SIZE = 128

sensor_size = tonic.datasets.NMNIST.sensor_size
frame_transform = tonic.transforms.ToFrame(sensor_size=sensor_size, time_window=10000)

train_data = tonic.datasets.NMNIST(root='./data', train=True, download=True)
test_data = tonic.datasets.NMNIST(root='./data', train=False, download=True)


