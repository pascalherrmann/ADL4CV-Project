# ADL4CV-Project

## Training / Experiments

To achieve reproducable & saved results, the following process is recommended:

* create a branch `run/<id>_<description>` and do the setup (e.g. change loss, architecture, etc.) you want to run
* edit the configuration file `src/config.py` and do the following adjustments:
    * point `DATA_DIR` to the `.tfrecords`-file you want to train the encoder with.
    * point `PICKLE_DIR` to the pickle-file of the pre-trained Generator-Model (the most common one is in `<your gdrive path>/tensorboards_shared/Training_Decoder_TF/00000-sgan-ffhq128-1gpu/network-snapshot-011489.pkl`)
    * point `GDRIVE_PATH` to the shared google drive folder where the logs should be saved (i.e., `<your gdrive path>/tensorboards_shared`)
* open jupyter notebook and run:


Clone project & checkout branch

```
!git clone https://adl4cv:GitHub111!@github.com/pascalherrmann/ADL4CV-Project
%cd ADL4CV-Project/src
!git checkout run/<your branch you want to run>
```

Import 1.x version of tensorflow & Connect Google Drive

```
%tensorflow_version 1.x
import tensorflow as tf
print(tf.__version__)

%load_ext autoreload
%autoreload 2

from google.colab import drive
drive.mount('/content/gdrive')
```

Run Training

```
!python train_encoder.py
```

#### Continuing Training

If you want to **continue** training from a **checkpoint**, do the following:

* checkout the `run/<...>` -Branch from where you want to continue
* in the config file, point `ENCODER_PICKLE_DIR` to the checkpoint-pickle you want to load (*in the future we might automatically check *)
