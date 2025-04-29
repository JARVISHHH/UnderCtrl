import keras
import keras_cv

from cldm.diffuser import ResBlock, SpatialTransformer, Upsample
from cldm.utils import timestep_embedding, PaddedConv2D, ZeroPaddedConv2D

import tensorflow as tf
import numpy as np

MAX_PROMPT_LENGTH = 77

# The copied ControlNet
class ControlNet(keras.Model):
    def __init__(
            self,
            img_height,
            img_width,
            download_weights=True, 
        ):

        super().__init__()

        # Time embedding
        self.time_embedding_model = tf.keras.Sequential([
            # keras.layers.Input((320,), name="timestep_embedding"),
            keras.layers.Dense(1280),
            keras.layers.Activation("swish"),
            keras.layers.Dense(1280),
        ])

        # Compute hint embedding. See cldm/cldm.py Line 147.
        self.hint_embedding_model = tf.keras.Sequential([
            PaddedConv2D(filters=16, kernel_size=3, strides=1, padding=1, name='hint_conv2d_1'),
            keras.layers.Activation("swish", name='hint_swish_1'),
            PaddedConv2D(filters=16, kernel_size=3, strides=1, padding=1, name='hint_conv2d_2'),
            keras.layers.Activation("swish", name='hint_swish_2'),
            PaddedConv2D(filters=32, kernel_size=3, strides=2, padding=1, name='hint_conv2d_3'),
            keras.layers.Activation("swish", name='hint_swish_3'),
            PaddedConv2D(filters=32, kernel_size=3, strides=1, padding=1, name='hint_conv2d_4'),
            keras.layers.Activation("swish", name='hint_swish_4'),
            PaddedConv2D(filters=96, kernel_size=3, strides=2, padding=1, name='hint_conv2d_5'),
            keras.layers.Activation("swish", name='hint_swish_5'),
            PaddedConv2D(filters=96, kernel_size=3, strides=1, padding=1, name='hint_conv2d_6'),
            keras.layers.Activation("swish", name='hint_swish_6'),
            PaddedConv2D(filters=256, kernel_size=3, strides=2, padding=1, name='hint_conv2d_7'),
            keras.layers.Activation("swish", name='hint_swish_7'),
            ZeroPaddedConv2D(filters=320, kernel_size=3, strides=1, padding=1, name='hint_zeroconv2d_8'),
        ])

        # self.layers is a reserved field
        self.control_layers = [
            # Downsampling flow
            PaddedConv2D(320, kernel_size=3, padding=1),
            ZeroPaddedConv2D(filters=320, kernel_size=1, strides=1, padding=0, name='control_zeroconv2d_1'),  # Control
            ### SD Encoder Block 1
            ResBlock(320),
            SpatialTransformer(8, 40, fully_connected=False),
            ZeroPaddedConv2D(filters=320, kernel_size=1, strides=1, padding=0, name='control_zeroconv2d_2'),  # Control
            ResBlock(320),
            SpatialTransformer(8, 40, fully_connected=False),
            ZeroPaddedConv2D(filters=320, kernel_size=1, strides=1, padding=0, name='control_zeroconv2d_3'),  # Control
            PaddedConv2D(320, 3, strides=2, padding=1),
            ZeroPaddedConv2D(filters=320, kernel_size=1, strides=1, padding=0, name='control_zeroconv2d_4'),  # Control
            ### SD Encoder Block 2
            ResBlock(640),
            SpatialTransformer(8, 80, fully_connected=False),
            ZeroPaddedConv2D(filters=640, kernel_size=1, strides=1, padding=0, name='control_zeroconv2d_5'),  # Control
            ResBlock(640),
            SpatialTransformer(8, 80, fully_connected=False),
            ZeroPaddedConv2D(filters=640, kernel_size=1, strides=1, padding=0, name='control_zeroconv2d_6'),  # Control
            PaddedConv2D(640, 3, strides=2, padding=1),
            ZeroPaddedConv2D(filters=640, kernel_size=1, strides=1, padding=0, name='control_zeroconv2d_7'),  # Control
            ### SD Encoder Block 3
            ResBlock(1280),
            SpatialTransformer(8, 160, fully_connected=False),
            ZeroPaddedConv2D(filters=1280, kernel_size=1, strides=1, padding=0, name='control_zeroconv2d_8'),  # Control
            ResBlock(1280),
            SpatialTransformer(8, 160, fully_connected=False),
            ZeroPaddedConv2D(filters=1280, kernel_size=1, strides=1, padding=0, name='control_zeroconv2d_9'),  # Control
            PaddedConv2D(1280, 3, strides=2, padding=1),
            ZeroPaddedConv2D(filters=1280, kernel_size=1, strides=1, padding=0, name='control_zeroconv2d_10'),  # Control
            ### SD Encoder Block
            ResBlock(1280),
            ZeroPaddedConv2D(filters=1280, kernel_size=1, strides=1, padding=0, name='control_zeroconv2d_11'),  # Control
            ResBlock(1280),
            ZeroPaddedConv2D(filters=1280, kernel_size=1, strides=1, padding=0, name='control_zeroconv2d_12'),  # Control
            # Middle flow
            ResBlock(1280),
            SpatialTransformer(8, 160, fully_connected=False),
            ResBlock(1280),
            ZeroPaddedConv2D(filters=1280, kernel_size=1, strides=1, padding=0, name='control_zeroconv2d_13'),  # Control
        ]

    def __call__(self, inputs):
        context, t_embed_input, latent, control = inputs
        t_emb = self.time_embedding_model(t_embed_input)
        guided_hint = self.hint_embedding_model(control)

        index = 0
        final_output = []

        # Downsampling flow
        x = self.control_layers[index](latent); index += 1; x += guided_hint
        final_output.append(self.control_layers[index](x)); index += 1
        ### SD Encoder Block 1
        for _ in range(2):
            x = self.control_layers[index]([x, t_emb]); index += 1
            x = self.control_layers[index]([x, context]); index += 1
            final_output.append(self.control_layers[index](x)); index += 1
        x = self.control_layers[index](x); index += 1
        final_output.append(self.control_layers[index](x)); index += 1
        ### SD Encoder Block 2
        for _ in range(2):
            x = self.control_layers[index]([x, t_emb]); index += 1
            x = self.control_layers[index]([x, context]); index += 1
            final_output.append(self.control_layers[index](x)); index += 1
        x = self.control_layers[index](x); index += 1
        final_output.append(self.control_layers[index](x)); index += 1
        ### SD Encoder Block 3
        for _ in range(2):
            x = self.control_layers[index]([x, t_emb]); index += 1
            x = self.control_layers[index]([x, context]); index += 1
            final_output.append(self.control_layers[index](x)); index += 1
        x = self.control_layers[index](x); index += 1
        final_output.append(self.control_layers[index](x)); index += 1
        ### SD Encoder Block
        for _ in range(2):
            x = self.control_layers[index]([x, t_emb]); index += 1
            final_output.append(self.control_layers[index](x)); index += 1

        # Middle flow
        x = self.control_layers[index]([x, t_emb]); index += 1
        x = self.control_layers[index]([x, context]); index += 1
        x = self.control_layers[index]([x, t_emb]); index += 1
        final_output.append(self.control_layers[index](x)); index += 1

        return final_output

# The locked UNet
class ControlledUnetModel(keras.Model):
    def __init__(
        self,
    ):
        super().__init__()

        # Time embedding
        self.time_embedding_model = tf.keras.Sequential([
            keras.layers.Dense(1280),
            keras.layers.Activation("swish"),
            keras.layers.Dense(1280),
        ])

        self.unet_layers = [
            # Downsampling flow
            PaddedConv2D(320, kernel_size=3, padding=1, trainable=False, name='lock_1'),
            ### SD Encoder Block 1
            ResBlock(320, trainable=False, name='lock_2'),
            SpatialTransformer(8, 40, fully_connected=False, trainable=False, name='lock_3'),
            ResBlock(320, trainable=False, name='lock_4'),
            SpatialTransformer(8, 40, fully_connected=False, trainable=False, name='lock_5'),
            PaddedConv2D(320, 3, strides=2, padding=1, trainable=False, name='lock_6'),
            ### SD Encoder Block 2
            ResBlock(640, trainable=False, name='lock_7'),
            SpatialTransformer(8, 80, fully_connected=False, trainable=False, name='lock_8'),
            ResBlock(640, trainable=False, name='lock_9'),
            SpatialTransformer(8, 80, fully_connected=False, trainable=False, name='lock_10'),
            PaddedConv2D(640, 3, strides=2, padding=1, trainable=False, name='lock_11'),
            ### SD Encoder Block 3
            ResBlock(1280, trainable=False, name='lock_12'),
            SpatialTransformer(8, 160, fully_connected=False, trainable=False, name='lock_13'),
            ResBlock(1280, trainable=False, name='lock_14'),
            SpatialTransformer(8, 160, fully_connected=False, trainable=False, name='lock_15'),
            PaddedConv2D(1280, 3, strides=2, padding=1, trainable=False, name='lock_16'),
            ### SD Encoder Block
            ResBlock(1280, trainable=False, name='lock_17'),
            ResBlock(1280, trainable=False, name='lock_18'),
            # Middle flow
            ResBlock(1280, trainable=False, name='lock_19'),
            SpatialTransformer(8, 160, fully_connected=False, trainable=False, name='lock_20'),
            ResBlock(1280, trainable=False, name='lock_21'),
            # Upsampling flow
            ### SD Decoder
            keras.layers.Concatenate(),
            ResBlock(1280),
            keras.layers.Concatenate(),
            ResBlock(1280),
            keras.layers.Concatenate(),
            ResBlock(1280),
            Upsample(1280),
            ### SD Decoder 3
            keras.layers.Concatenate(),
            ResBlock(1280),
            SpatialTransformer(8, 160, fully_connected=False),
            keras.layers.Concatenate(),
            ResBlock(1280),
            SpatialTransformer(8, 160, fully_connected=False),
            keras.layers.Concatenate(),
            ResBlock(1280),
            SpatialTransformer(8, 160, fully_connected=False),
            Upsample(1280),
            ### SD Decoder 2
            keras.layers.Concatenate(),
            ResBlock(640),
            SpatialTransformer(8, 80, fully_connected=False),
            keras.layers.Concatenate(),
            ResBlock(640),
            SpatialTransformer(8, 80, fully_connected=False),
            keras.layers.Concatenate(),
            ResBlock(640),
            SpatialTransformer(8, 80, fully_connected=False),
            Upsample(640),
            ### SD Decoder 1
            keras.layers.Concatenate(),
            ResBlock(320),
            SpatialTransformer(8, 40, fully_connected=False),
            keras.layers.Concatenate(),
            ResBlock(320),
            SpatialTransformer(8, 40, fully_connected=False),
            keras.layers.Concatenate(),
            ResBlock(320),
            SpatialTransformer(8, 40, fully_connected=False),
            # Exit flow
            keras.layers.GroupNormalization(epsilon=1e-5),
            keras.layers.Activation("swish"),
            PaddedConv2D(4, kernel_size=3, padding=1)
        ]

    def __call__(self, inputs):
        context, t_embed_input, latent, control = inputs
        t_emb = self.time_embedding_model(t_embed_input)

        index = 0
        output = []

        # Downsampling flow
        x = self.unet_layers[index](latent); index += 1
        output.append(x)
        ### SD Encoder Block 1
        for _ in range(2):
            x = self.unet_layers[index]([x, t_emb]); index += 1
            x = self.unet_layers[index]([x, context]); index += 1
            output.append(x)
        x = self.unet_layers[index](x); index += 1
        output.append(x)
        ### SD Encoder Block 2
        for _ in range(2):
            x = self.unet_layers[index]([x, t_emb]); index += 1
            x = self.unet_layers[index]([x, context]); index += 1
            output.append(x)
        x = self.unet_layers[index](x); index += 1
        output.append(x)
        ### SD Encoder Block 3
        for _ in range(2):
            x = self.unet_layers[index]([x, t_emb]); index += 1
            x = self.unet_layers[index]([x, context]); index += 1
            output.append(x)
        x = self.unet_layers[index](x); index += 1
        output.append(x)
        ### SD Encoder Block
        for _ in range(2):
            x = self.unet_layers[index]([x, t_emb]); index += 1
            output.append(x)

        # Middle flow
        x = self.unet_layers[index]([x, t_emb]); index += 1
        x = self.unet_layers[index]([x, context]); index += 1
        x = self.unet_layers[index]([x, t_emb]); index += 1

        x += control.pop()

        # Upsampling flow
        ### SD Decoder
        for _ in range(3):
            x = self.unet_layers[index]([x, output.pop() + control.pop()]); index += 1
            x = self.unet_layers[index]([x, t_emb]); index += 1
        x = self.unet_layers[index](x); index += 1
        ### SD Decoder 3
        for _ in range(3):
            x = self.unet_layers[index]([x, output.pop() + control.pop()]); index += 1
            x = self.unet_layers[index]([x, t_emb]); index += 1
            x = self.unet_layers[index]([x, context]); index += 1
        x = self.unet_layers[index](x); index += 1
        ### SD Decoder 2
        for _ in range(3):
            x = self.unet_layers[index]([x, output.pop() + control.pop()]); index += 1
            x = self.unet_layers[index]([x, t_emb]); index += 1
            x = self.unet_layers[index]([x, context]); index += 1
        x = self.unet_layers[index](x); index += 1
        ### SD Decoder 1
        for _ in range(3):
            x = self.unet_layers[index]([x, output.pop() + control.pop()]); index += 1
            x = self.unet_layers[index]([x, t_emb]); index += 1
            x = self.unet_layers[index]([x, context]); index += 1

        # Exit flow
        x = self.unet_layers[index](x); index += 1
        x = self.unet_layers[index](x); index += 1
        x = self.unet_layers[index](x); index += 1

        return x
        

class ControlSDB(keras_cv.models.StableDiffusion):
    def __init__(
        self,
        optimizer,
        img_height=64,
        img_width=64,
    ):
        super().__init__(img_height, img_width)

        self.noise_scheduler = keras_cv.models.stable_diffusion.NoiseScheduler()
        self.max_grad_norm = 1.0

        self.control_model = ControlNet(img_height, img_width)
        self.control_scales = [1.0] * 13

        self.diffuser = ControlledUnetModel()

        self.optimizer = optimizer


    def build(self, input_shape):
        batch_size = input_shape[0] if input_shape[0] is not None else 1
        img_height = input_shape[1]
        img_width = input_shape[2]

        # Prepare dummy inputs manually
        latent_shape = (batch_size, img_height // 8, img_width // 8, 4)
        latents = tf.zeros(latent_shape, dtype=tf.float32)

        context_shape = (batch_size, MAX_PROMPT_LENGTH, 768) # TODO
        dummy_context = tf.zeros(context_shape, dtype=tf.float32)

        timestep_shape = (batch_size,)
        dummy_timesteps = tf.zeros(timestep_shape, dtype=tf.int32)
        timestep_emb = timestep_embedding(dummy_timesteps)

        control_shape = (batch_size, img_height, img_width, 3)
        controls = tf.zeros(control_shape, dtype=tf.float32)

        # Build control model
        control_outputs = self.control_model([dummy_context, timestep_emb, latents, controls])
        self.control_model.built = True
        control_outputs = [c * scale for c, scale in zip(control_outputs, self.control_scales)]

        # Build diffusion model
        _ = self.diffuser([dummy_context, timestep_emb, latents, control_outputs])
        self.diffuser.built = True

        self.control_model.summary()
        self.diffuser.summary()

    def train_step(self, inputs):
        latents, encoded_text, controls = self.get_input(inputs)
        batch_size = tf.shape(latents)[0]

        with tf.GradientTape() as tape:
            noise = tf.random.normal(shape=tf.shape(latents), dtype=latents.dtype)
            timesteps = tf.random.stateless_uniform(
                shape=(batch_size,),
                seed=(0, 0),
                minval=0,
                maxval=self.noise_scheduler.train_timesteps,
                dtype=tf.int32
            )
            noisy_latents = self.noise_scheduler.add_noise(
                latents, noise, timesteps
            )
            target = noise

            # Check this
            timestep_embedding_not_trainnable = tf.stop_gradient(timestep_embedding(timesteps))
            timestep_embedding_trainnable = timestep_embedding(timesteps)

            if isinstance(encoded_text, list):
                encoded_text = tf.stack(tf.squeeze(encoded_text, axis=1), axis=0)
            else:
                encoded_text = tf.convert_to_tensor(encoded_text)
            
            control = self.control_model([encoded_text, timestep_embedding_trainnable, noisy_latents, controls])
            control = [c * scale for c, scale in zip(control, self.control_scales)]

            eps = self.diffuser([encoded_text, timestep_embedding_not_trainnable, noisy_latents, control])

            loss_fn = tf.keras.losses.MeanSquaredError(
                reduction='sum_over_batch_size',
                name='mean_squared_error'
            )
            loss = loss_fn(target, eps)
            loss = tf.cast(loss, tf.float32)
        
        trainable_vars = self.control_model.trainable_variables + self.diffuser.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        gradients = [tf.clip_by_norm(g, self.max_grad_norm) for g in gradients]
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        return { 'loss': loss }

    def predict(self, inputs):
        latents, encoded_text, control = self.get_input(inputs)
        batch_size = tf.shape(latents)[0]
        
        timesteps = np.random.randint(0, self.noise_scheduler.train_timesteps, (batch_size,))
        
        for t in timesteps:
            t = tf.cast(t, 'int32')
            
            alpha_prod_t = tf.maximum(self.noise_scheduler.alphas_cumprod[t], 1e-6)
            alpha_prod_t_prev = tf.maximum(
                self.noise_scheduler.alphas_cumprod[t-1] if t > 0 else 1.0,
                1e-6
            )
            
            t_embed = tf.stop_gradient(timestep_embedding(tf.repeat(t, batch_size)))
            
            if isinstance(encoded_text, list):
                encoded_text = tf.stack(tf.squeeze(encoded_text, axis=1), axis=0)
            else:
                encoded_text = tf.convert_to_tensor(encoded_text)

            control = self.control_model([encoded_text, t_embed, latents, controls])
            control = [c * scale for c, scale in zip(control, self.control_scales)]
       
            eps = self.diffuser([encoded_text, t_embed, latents, control])
            
            # Copying part of noise_scheduler's step function because it's incompatible with the keras_cv version
            beta_prod_t = 1 - alpha_prod_t
            pred_original = (latents - beta_prod_t**0.5 * eps) / tf.maximum(alpha_prod_t**0.5, 1e-3)
            
            variance = (beta_prod_t / alpha_prod_t) * (1 - alpha_prod_t_prev / alpha_prod_t)
            noise = tf.random.normal(tf.shape(latents)) if t > 0 else 0.0
            temp = alpha_prod_t_prev**0.5 * pred_original + tf.sqrt(variance) * noise

            if tf.reduce_any(tf.math.is_nan(temp)):
                print(f"NaN detected at t={t} - resetting latents")
                break
                
            latents = temp
        
        images = self.decoder(latents)
        images = tf.clip_by_value(images, -1.0, 1.0)
        return (images + 1.0) * 127.5

    def get_input(self, inputs):
        # TODO: should images be rearranged? from (b h w c) to (b c h w)?
        # jpg refer to get_input() in https://github.com/lllyasviel/ControlNet/blob/main/ldm/models/diffusion/ddpm.py#L419
        images = inputs["jpg"]
        latents = self.image_encoder(images)
        # condition/prompt/txt refer to get_input() in https://github.com/lllyasviel/ControlNet/blob/main/ldm/models/diffusion/ddpm.py#L767
        encoded_text = inputs["txt"]
        # control refer to get_input() in https://github.com/lllyasviel/ControlNet/blob/main/cldm/cldm.py#L318
        controls = inputs["hint"]

        return latents, encoded_text, controls