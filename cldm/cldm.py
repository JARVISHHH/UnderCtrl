from keras_cv.src.backend import keras
from keras_cv.src.models.stable_diffusion.padded_conv2d import PaddedConv2D

from cldm.diffuser import ResBlock, SpatialTransformer, Upsample
from utils import ZeroPaddedConv2D

# The copied ControlNet
class ControlNet(keras.Model):
    def __init__(
            self,
            img_height,
            img_width,
            max_text_length,
            hint_image_size,
            name=None,
            download_weights=True, 
        ):

        # Inputs
        context = keras.layers.Input((max_text_length, 768), name="context")
        t_embed_input = keras.layers.Input((320,), name="timestep_embedding")
        latent = keras.layers.Input(
            (img_height // 8, img_width // 8, 4), name="latent"
        )
        hint = keras.layers.Input(hint_image_size, name="hint")

        # Time embedding
        t_emb = keras.layers.Dense(1280)(t_embed_input)
        t_emb = keras.layers.Activation("swish")(t_emb)
        t_emb = keras.layers.Dense(1280)(t_emb)

        # Compute hint embedding. See cldm/cldm.py Line 147.
        guided_hint = PaddedConv2D(filters=16, kernel_size=3, strides=1, padding=1)(hint)
        guided_hint = keras.layers.Activation("swish")(guided_hint)
        guided_hint = PaddedConv2D(filters=16, kernel_size=3, strides=1, padding=1)(guided_hint)
        guided_hint = keras.layers.Activation("swish")(guided_hint)
        guided_hint = PaddedConv2D(filters=32, kernel_size=3, strides=2, padding=1)(guided_hint)
        guided_hint = keras.layers.Activation("swish")(guided_hint)
        guided_hint = PaddedConv2D(filters=32, kernel_size=3, strides=1, padding=1)(guided_hint)
        guided_hint = keras.layers.Activation("swish")(guided_hint)
        guided_hint = PaddedConv2D(filters=96, kernel_size=3, strides=2, padding=1)(guided_hint)
        guided_hint = keras.layers.Activation("swish")(guided_hint)
        guided_hint = PaddedConv2D(filters=96, kernel_size=3, strides=1, padding=1)(guided_hint)
        guided_hint = keras.layers.Activation("swish")(guided_hint)
        guided_hint = PaddedConv2D(filters=256, kernel_size=3, strides=2, padding=1)(guided_hint)
        guided_hint = keras.layers.Activation("swish")(guided_hint)
        guided_hint = ZeroPaddedConv2D(filters=320, kernel_size=3, strides=2, padding=1)(guided_hint)

        # Final results with control
        final_outputs = []

        # Downsampling flow
        outputs = []
        x = PaddedConv2D(320, kernel_size=3, padding=1)(latent)
        x += guided_hint
        outputs.append(x)
        final_outputs.append(ZeroPaddedConv2D(filters=320, kernel_size=1, strides=1, padding=0)(x))
        ### SD Encoder Block 1   
        for _ in range(2):
            x = ResBlock(320)([x, t_emb])
            x = SpatialTransformer(8, 40, fully_connected=False)([x, context])  # channels = 8 * 40
            outputs.append(x)
            final_outputs.append(ZeroPaddedConv2D(filters=320, kernel_size=1, strides=1, padding=0)(x))
        x = PaddedConv2D(320, 3, strides=2, padding=1)(x)  # Downsample 2x
        outputs.append(x)
        final_outputs.append(ZeroPaddedConv2D(filters=320, kernel_size=1, strides=1, padding=0)(x))
        ### SD Encoder Block 2
        for _ in range(2):
            x = ResBlock(640)([x, t_emb])
            x = SpatialTransformer(8, 80, fully_connected=False)([x, context])  # channels = 8 * 80
            outputs.append(x)
            final_outputs.append(ZeroPaddedConv2D(filters=640, kernel_size=1, strides=1, padding=0)(x))
        x = PaddedConv2D(640, 3, strides=2, padding=1)(x)  # Downsample 2x
        outputs.append(x)
        final_outputs.append(ZeroPaddedConv2D(filters=640, kernel_size=1, strides=1, padding=0)(x))
        ### SD Encoder Block 3
        for _ in range(2):
            x = ResBlock(1280)([x, t_emb])
            x = SpatialTransformer(8, 160, fully_connected=False)([x, context])  # channels = 8 * 160
            outputs.append(x)
            final_outputs.append(ZeroPaddedConv2D(filters=1280, kernel_size=1, strides=1, padding=0)(x))
        x = PaddedConv2D(1280, 3, strides=2, padding=1)(x)  # Downsample 2x
        outputs.append(x)
        final_outputs.append(ZeroPaddedConv2D(filters=1280, kernel_size=1, strides=1, padding=0)(x))
        ### SD Encoder Block
        for _ in range(2):
            x = ResBlock(1280)([x, t_emb])
            outputs.append(x)
            final_outputs.append(ZeroPaddedConv2D(filters=1280, kernel_size=1, strides=1, padding=0)(x))

        # Middle flow
        x = ResBlock(1280)([x, t_emb])
        x = SpatialTransformer(8, 160, fully_connected=False)([x, context])  # channels = 8 * 160
        x = ResBlock(1280)([x, t_emb])
        final_outputs.append(ZeroPaddedConv2D(filters=1280, kernel_size=1, strides=1, padding=0)(x))

        super().__init__([latent, t_embed_input, context], final_outputs, name=name)


# The locked UNet
class ControlledUnetModel(keras.Model):
    def __init__(
        self,
        img_height,
        img_width,
        max_text_length,
        name=None,
        control=None,
        download_weights=True, 
    ):
        context = keras.layers.Input((max_text_length, 768), name="context")
        t_embed_input = keras.layers.Input((320,), name="timestep_embedding")
        latent = keras.layers.Input(
            (img_height // 8, img_width // 8, 4), name="latent"
        )

        t_emb = keras.layers.Dense(1280)(t_embed_input)
        t_emb = keras.layers.Activation("swish")(t_emb)
        t_emb = keras.layers.Dense(1280)(t_emb)

        # Downsampling flow
        outputs = []
        x = PaddedConv2D(320, kernel_size=3, padding=1)(latent)
        outputs.append(x)
        ### SD Encoder Block 1
        for _ in range(2):
            x = ResBlock(320)([x, t_emb])
            x = SpatialTransformer(8, 40, fully_connected=False)([x, context])
            outputs.append(x)
        x = PaddedConv2D(320, 3, strides=2, padding=1)(x)  # Downsample 2x
        outputs.append(x)
        ### SD Encoder Block 2
        for _ in range(2):
            x = ResBlock(640)([x, t_emb])
            x = SpatialTransformer(8, 80, fully_connected=False)([x, context])
            outputs.append(x)
        x = PaddedConv2D(640, 3, strides=2, padding=1)(x)  # Downsample 2x
        outputs.append(x)
        ### SD Encoder Block 3
        for _ in range(2):
            x = ResBlock(1280)([x, t_emb])
            x = SpatialTransformer(8, 160, fully_connected=False)([x, context])
            outputs.append(x)
        x = PaddedConv2D(1280, 3, strides=2, padding=1)(x)  # Downsample 2x
        outputs.append(x)
        ### SD Encoder Block
        for _ in range(2):
            x = ResBlock(1280)([x, t_emb])
            outputs.append(x)

        # Middle flow
        x = ResBlock(1280)([x, t_emb])
        x = SpatialTransformer(8, 160, fully_connected=False)([x, context])
        x = ResBlock(1280)([x, t_emb])

        x += control.pop()

        # Upsampling flow
        self.trainable_layers = []
        ### SD Decoder
        for _ in range(3):
            x = keras.layers.Concatenate()([x, outputs.pop() + control.pop()])
            self.trainable_layers.append(x)
            x = ResBlock(1280)([x, t_emb])
            self.trainable_layers.append(x)
        x = Upsample(1280)(x)
        self.trainable_layers.append(x)
        ### SD Decoder 3
        for _ in range(3):
            x = keras.layers.Concatenate()([x, outputs.pop() + control.pop()])
            self.trainable_layers.append(x)
            x = ResBlock(1280)([x, t_emb])
            self.trainable_layers.append(x)
            x = SpatialTransformer(8, 160, fully_connected=False)([x, context])
            self.trainable_layers.append(x)
        x = Upsample(1280)(x)
        self.trainable_layers.append(x)
        ### SD Decoder 2
        for _ in range(3):
            x = keras.layers.Concatenate()([x, outputs.pop() + control.pop()])
            self.trainable_layers.append(x)
            x = ResBlock(640)([x, t_emb])
            self.trainable_layers.append(x)
            x = SpatialTransformer(8, 80, fully_connected=False)([x, context])
            self.trainable_layers.append(x)
        x = Upsample(640)(x)
        self.trainable_layers.append(x)
        ### SD Decoder 1
        for _ in range(3):
            x = keras.layers.Concatenate()([x, outputs.pop() + control.pop()])
            self.trainable_layers.append(x)
            x = ResBlock(320)([x, t_emb])
            self.trainable_layers.append(x)
            x = SpatialTransformer(8, 40, fully_connected=False)([x, context])
            self.trainable_layers.append(x)

        # Exit flow
        x = keras.layers.GroupNormalization(epsilon=1e-5)(x)
        self.trainable_layers.append(x)
        x = keras.layers.Activation("swish")(x)
        self.trainable_layers.append(x)
        output = PaddedConv2D(4, kernel_size=3, padding=1)(x)
        self.trainable_layers.append(output)

        for layer in self.layers:
            layer.trainable = False
        for layer in self.trainable_layers:
            layer.trainable = True

        super().__init__([latent, t_embed_input, context], output, name=name)

        # if download_weights:
        #     diffusion_model_weights_fpath = keras.utils.get_file(
        #         origin="https://huggingface.co/fchollet/stable-diffusion/resolve/main/kcv_diffusion_model.h5",  # noqa: E501
        #         file_hash="8799ff9763de13d7f30a683d653018e114ed24a6a819667da4f5ee10f9e805fe",  # noqa: E501
        #     )
        #     self.load_weights(diffusion_model_weights_fpath)


# Wrapper
class ControlLDM(keras.Model):
    def __init__(
        self,
        diffusion_model,
        control_model,
        vae,
        noise_scheduler,
        use_mixed_precision=False,
        max_grad_norm=1.0,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.diffusion_model = diffusion_model
        self.control_model = control_model
        self.vae = vae
        self.noise_scheduler = noise_scheduler
        self.max_grad_norm = max_grad_norm

        self.use_mixed_precision = use_mixed_precision
        self.vae.trainable = False

        self.control_key = "hint"

    # def train_step(self, inputs):
    #     images = inputs["images"]
    #     encoded_text = inputs["encoded_text"]
    #     control = inputs[self.control_key]
    #     batch_size = tf.shape(images)[0]

    #     with tf.GradientTape() as tape:
    #         # Project image into the latent space and sample from it.
    #         latents = self.sample_from_encoder_outputs(self.vae(images, training=False))
    #         # Know more about the magic number here:
    #         # https://keras.io/examples/generative/fine_tune_via_textual_inversion/
    #         latents = latents * 0.18215

    #         # Sample noise that we'll add to the latents.
    #         noise = tf.random.normal(tf.shape(latents))

    #         # Sample a random timestep for each image.
    #         timesteps = np.random.randint(
    #             0, self.noise_scheduler.train_timesteps, (batch_size,)
    #         )

    #         # Add noise to the latents according to the noise magnitude at each timestep
    #         # (this is the forward diffusion process).
    #         noisy_latents = self.noise_scheduler.add_noise(
    #             tf.cast(latents, noise.dtype), noise, timesteps
    #         )

    #         # Get the target for loss depending on the prediction type
    #         # just the sampled noise for now.
    #         target = noise  # noise_schedule.predict_epsilon == True

    #         # Predict the noise residual and compute loss.
    #         timestep_embedding = tf.map_fn(
    #             lambda t: self.get_timestep_embedding(t), timesteps, dtype=tf.float32
    #         )
    #         timestep_embedding = tf.squeeze(timestep_embedding, 1)
    #         model_pred = self.diffusion_model(
    #             [noisy_latents, timestep_embedding, encoded_text], training=True
    #         )
    #         loss = self.compiled_loss(target, model_pred)
    #         if self.use_mixed_precision:
    #             loss = self.optimizer.get_scaled_loss(loss)

    #     # Update parameters of the diffusion model.
    #     trainable_vars = self.diffusion_model.trainable_variables
    #     gradients = tape.gradient(loss, trainable_vars)
    #     if self.use_mixed_precision:
    #         gradients = self.optimizer.get_unscaled_gradients(gradients)
    #     gradients = [tf.clip_by_norm(g, self.max_grad_norm) for g in gradients]
    #     self.optimizer.apply_gradients(zip(gradients, trainable_vars))

    #     return {m.name: m.result() for m in self.metrics}

    # def get_timestep_embedding(self, timestep, dim=320, max_period=10000):
    #     half = dim // 2
    #     log_max_period = tf.math.log(tf.cast(max_period, tf.float32))
    #     freqs = tf.math.exp(
    #         -log_max_period * tf.range(0, half, dtype=tf.float32) / half
    #     )
    #     args = tf.convert_to_tensor([timestep], dtype=tf.float32) * freqs
    #     embedding = tf.concat([tf.math.cos(args), tf.math.sin(args)], 0)
    #     embedding = tf.reshape(embedding, [1, -1])
    #     return embedding

    # def sample_from_encoder_outputs(self, outputs):
    #     mean, logvar = tf.split(outputs, 2, axis=-1)
    #     logvar = tf.clip_by_value(logvar, -30.0, 20.0)
    #     std = tf.exp(0.5 * logvar)
    #     sample = tf.random.normal(tf.shape(mean), dtype=mean.dtype)
    #     return mean + std * sample

    def save_weights(self, filepath, overwrite=True, save_format=None, options=None):
        self.control_model.save_weights(
            filepath=filepath,
            overwrite=overwrite,
            save_format=save_format,
            options=options,
        )