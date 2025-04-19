# UnderCtrl

libs for loading and preprocessing data

- datasets 



## Dependencies

Reuse stable diffusion in keras_cv.

[Source Code](https://github.com/keras-team/keras-cv/tree/master/keras_cv/src/models/stable_diffusion)

[Fine-tuning Example](https://keras.io/examples/generative/finetune_stable_diffusion/)



## Model architecture

The `ControlNet` class and `ControlledUnetModel` class are implemented based on `DiffusionModel` class in [source code](https://github.com/keras-team/keras-cv/blob/master/keras_cv/src/models/stable_diffusion/diffusion_model.py#L22). These 2 classes correspond to classes with the same name in [original implementation](https://github.com/lllyasviel/ControlNet/blob/main/cldm/cldm.py) of ControlNet.



## Todo List

- [ ] wrapper for stable diffusion (class `ControlledUnetModel`)
- [ ] load pretrained weights into our customized layers
- [ ] ...