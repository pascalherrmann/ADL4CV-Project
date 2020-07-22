# Metrics

### Run Instructions

##### Clone Repository

```python
!git clone https://github.com/pascalherrmann/ADL4CV-Project
%cd /content/ADL4CV-Project/src
```

##### Import TensorFlow, Mount gDrive

```python
%tensorflow_version 1.x
import tensorflow as tf
print(tf.__version__)

%load_ext autoreload
%autoreload 2

from google.colab import drive
drive.mount('/content/gdrive')
```

##### Imports

```python
import pickle

from dnnlib import EasyDict
import dnnlib.tflib as tflib
import config

from metrics import metric_base
```

##### Run Metrics

```python
model_path = "/content/gdrive/My Drive/Public/tensorboards_shared/run/t/58_rignet_fixed_pose_only_larger_adv_scaling/snapshots/network-snapshot-01040128.pkl"
dataset_args = EasyDict(tfrecord_dir=config.TEST_DATA_DIR, resolution = config.RESOLUTION)
metrics = metric_base.MetricGroup([metric_base.lm_hd, metric_base.cism, metric_base.fid50k, metric_base.gallery])

metrics.run(model_path, num_gpus=1, dataset_args=dataset_args)
```
