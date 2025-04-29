# UnderCtrl

## Environment for Oscar
```
python3 -m venv ~/controlnet-env
source ~/controlnet-env/bin/activate
pip install -r requirements.txt
```

## Dependencies

Reuse stable diffusion in keras_cv.

[Source Code](https://github.com/keras-team/keras-cv/tree/master/keras_cv/src/models/stable_diffusion)

[Fine-tuning Example](https://keras.io/examples/generative/finetune_stable_diffusion/)



## Model architecture

The `ControlNet` class and `ControlledUnetModel` class are implemented based on `DiffusionModel` class in [source code](https://github.com/keras-team/keras-cv/blob/master/keras_cv/src/models/stable_diffusion/diffusion_model.py#L22). These 2 classes correspond to classes with the same name in [original implementation](https://github.com/lllyasviel/ControlNet/blob/main/cldm/cldm.py) of ControlNet.

`ControlSDB` is now the wrapper.



## Todo List

in a timely manner

- [ ] make training process runnable
- [ ] load pretrained weights into our customized layers
- [ ] add predict in `ControlSDB`
- [ ] test on toy dataset
- [ ] ...