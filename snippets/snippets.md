
##### Load Model & Show Sample Image

```python
import dnnlib
import dnnlib.tflib as tflib
import config
import pickle
import PIL
import numpy as np

tflib.init_tf()
with open(config.PICKLE_DIR,"rb") as f:
        _G, _D, Gs = pickle.load(f)

fmt = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)

rnd = np.random.RandomState(907236921)
latent_vector1 = rnd.randn(1, Gs.input_shape[1])
images = Gs.run(latent_vector1, None, truncation_psi=1, randomize_noise=False, output_transform=fmt)
PIL.Image.fromarray(images[0])
````


