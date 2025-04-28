# train only: python main.py --dataset <dataset> --epochs <epochs> --train --save_imgs
# resume training: python main.py --dataset <dataset> --epochs <epochs> --resume --load_epoch <load_epoch> --train --save_imgs
# test only: python main.py --dataset <dataset> --resume --load_epoch <load_epoch> --test
# keep batch size same for train and test

import argparse
import tensorflow as tf
from cldm.cldm import ControlSDB
import keras_cv
import keras
import numpy as np
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
import tensorflow_probability as tfp
from transformers import TFCLIPModel, CLIPProcessor
from PIL import Image
import os

import matplotlib.pyplot as plt

inception_model = InceptionV3(include_top=False, weights='imagenet', pooling='avg')
clip_tokenizer = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32", use_fast=False)
clip_model = TFCLIPModel.from_pretrained("openai/clip-vit-base-patch32")

def parse_args():
    parser = argparse.ArgumentParser(description="Train and Test ControlNet")
    parser.add_argument('--dataset', type=str, default='fill50k', choices=['fill50k', 'facesynthetics'])
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--save_dir', type=str, default='./checkpoints', help='Directory to save checkpoints and images')
    parser.add_argument('--img_size', type=int, default=256)
    parser.add_argument('--resume', action='store_true') # Set to True to resume training
    parser.add_argument('--load_dir', type=str, default='./checkpoints')
    parser.add_argument('--load_epoch', type=int, default=1)
    parser.add_argument('--train', action='store_true', help='Test the model') # Set to True to train the model
    parser.add_argument('--save_imgs', action='store_true', help='Save images') # Set to True to save images
    parser.add_argument('--test', action='store_true', help='Test the model') # Set to True to test the model
    return parser.parse_args()

def main():
    args = parse_args()

    os.makedirs(args.save_dir, exist_ok=True) # if exists, do nothing

    if args.dataset == 'fill50k':
        from test_imgs import fill50k
        train_dataset, test_dataset, dataset_length = fill50k.get_dataset(args.batch_size, args.img_size)
    else:
        from test_imgs import facesynthetics
        train_dataset, test_dataset, dataset_length = facesynthetics.get_dataset(batch_size=args.batch_size, img_size=args.img_size)

    model = ControlSDB(optimizer=tf.keras.optimizers.Adam(learning_rate=args.lr), img_height=args.img_size, img_width=args.img_size)

    # build model
    dummy_input_shape = (args.batch_size, args.img_size, args.img_size, 3)
    model.build(dummy_input_shape)

    # load weights if available
    if args.resume:
        print("Loading weights...")
        try:
            model.control_model.load_weights(f"{args.load_dir}/controlnet_epoch_{args.load_epoch}.weights.h5")
            model.diffuser.load_weights(f"{args.load_dir}/unet_epoch_{args.load_epoch}.weights.h5")
            print("Loaded weights successfully.")
        except Exception as e:
            print("Failed to load weights:", e)
    else:
        print("Loading original weights...")
        try:
            file = keras.utils.get_file(
                                    origin="https://huggingface.co/fchollet/stable-diffusion/resolve/main/kcv_diffusion_model.h5",  # noqa: E501
                                    file_hash="8799ff9763de13d7f30a683d653018e114ed24a6a819667da4f5ee10f9e805fe",  # noqa: E501
            )
            model.control_model.load_weights(file, by_name=True, skip_mismatch=True)
            model.diffuser.load_weights(file, by_name=True, skip_mismatch=True)
            print("Loaded weights successfully.")
        except Exception as e:
            print("Failed to load weights:", e)
            print("Train from scratch.")

    # for epoch in range(args.epochs):
    #     for batch in train_dataset.take(1):
    #         losses.append(model.train_step(batch)['loss'])

    if args.train:
        print("----------Start Training----------")
        losses = []

        # Uncomment the following lines to test training the model with a single batch
        # for epoch in range(1):
        #     for batch in train_dataset.take(1):
        #         losses.append(model.train_step(batch)['loss'])
        #         print(f"Epoch {epoch+1}/{args.epochs}, Batch Loss: {losses[-1]:.6f}")

        start_epoch = args.load_epoch if args.resume else 0
        for epoch in range(start_epoch, start_epoch+args.epochs):
            epoch_loss = 0
            batch_num = 1
            for batch in train_dataset:
                loss = model.train_step(batch)
                epoch_loss += loss['loss']
                print(f"Epoch {epoch+1}/{args.epochs}, Batch {batch_num}/{dataset_length // args.batch_size} Loss: {loss['loss']:.6f}")
                batch_num += 1
            avg_epoch_loss = epoch_loss / len(train_dataset)
            losses.append(avg_epoch_loss)
            print(f"Epoch {epoch+1}/{args.epochs}, Loss: {avg_epoch_loss:.6f}")
            
            # Save the model weights after each epoch
            try:
                model.control_model.save_weights(f"{args.save_dir}/controlnet_epoch_{epoch+1}.weights.h5")
                model.diffuser.save_weights(f"{args.save_dir}/unet_epoch_{epoch+1}.weights.h5")
                print(f"Saved weights for epoch {epoch+1}.")
            except Exception as e:
                print("Failed to save weights:", e)
            # early stopping
            if avg_epoch_loss < 0.01:
                print("Early stopping...")
                break
        
        print("----------Finish Training----------")

    captions = [data['txt'] for data in test_dataset]
    captions = []
    for batch in test_dataset:
        captions.extend(batch['txt'].numpy().tolist())
    captions = [c.decode('utf-8') if isinstance(c, bytes) else c for c in captions]
    generated_images_sd = []
    stable_diffusion = keras_cv.models.StableDiffusion(img_width=args.img_size, img_height=args.img_size)

    if args.save_imgs:
        epochs = list(range(1, len(losses) + 1))
        plt.plot(epochs, losses, label='Loss')
        plt.title('Training Loss over Epochs')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.savefig(f"{args.save_dir}/loss_curve.png")
        plt.clf()
        plt.cla()
        plt.close('all')

    generated_images_controlnet = []
    for batch in test_dataset.take(1):
        image = model.predict(batch)
        generated_images_controlnet.append(image)
    generated_images_controlnet = model.predict(test_dataset)
    # print(generated_images_controlnet)

    # save_images(generated_images_sd, save_dir="outputs/sd", prefix="sd")
    # save_images(generated_images_controlnet, save_dir="outputs/controlnet", prefix="controlnet")

    if args.test:
        print("----------Start Testing----------")
        captions = [data['txt'] for data in test_dataset]
        generated_images_sd = []
        stable_diffusion = keras_cv.models.StableDiffusion(img_width=args.img_size, img_height=args.img_size)

        for caption in captions:
            generated_images_sd.append(stable_diffusion.text_to_image(caption, batch_size=args.batch_size))

        generated_images_controlnet = model.predict(test_dataset)

        clip_score_sd = calculate_clip_score(generated_images_sd, captions)
        clip_score_controlnet = calculate_clip_score(generated_images_controlnet, captions)
        print("CLIP Score:")
        print("Stable Diffusion: " + str(clip_score_sd))
        print("Control Net: " + str(clip_score_controlnet))

        real_images = [data['jpg'] for data in test_dataset]
        fid_score_sd = calculate_fid_score(real_images, generated_images_sd)
        fid_score_controlnet = calculate_fid_score(real_images, generated_images_controlnet)
        print("FID Score:")
        print("Stable Diffusion: " + str(fid_score_sd))
        print("Control Net: " + str(fid_score_controlnet))
        print("----------Finish Testing----------")

def resize_images(images, target_size=(299, 299)):
    images = tf.convert_to_tensor(images, dtype=tf.float32)
    resized = tf.image.resize(images, size=target_size, method='bilinear')
    return resized

def calculate_clip_score(images, captions):
    if isinstance(images, tf.Tensor):
        images = images.numpy()

    inputs = clip_tokenizer(
        text=captions,
        images=images,
        return_tensors="tf",  # Return TensorFlow tensors
        padding=True,
        truncation=True
    )

    outputs = clip_model(**inputs)

    image_embeddings = outputs.image_embeds
    text_embeddings = outputs.text_embeds

    # Normalize embeddings
    image_embeddings = tf.math.l2_normalize(image_embeddings, axis=1)
    text_embeddings = tf.math.l2_normalize(text_embeddings, axis=1)

    similarity = tf.reduce_sum(image_embeddings * text_embeddings, axis=1)

    return tf.reduce_mean(similarity).numpy().item()
    
def get_activations(images: tf.Tensor, batch_size=32):
    # Assuming InceptionV3 accepts 299x299 images
    images_resized = tf.image.resize(images, (299, 299))
    # Assuming images are within [-1, 1]
    images_preprocessed = preprocess_input(images_resized * 255.0)

    activations = inception_model(images_preprocessed, training=False)
    return activations

def calculate_fid_score(real_images, generated_images):
    real_images = tf.concat(real_images, axis=0)
    generated_images = tf.concat(generated_images, axis=0)

    # d^2 = ||mu_1 – mu_2||^2 + Tr(c_1 + c_2 – 2*sqrt(c_1*c_2))
    act_1 = get_activations(real_images)
    act_2 = get_activations(generated_images)

    # Mean and covariance
    mu_1 = tf.reduce_mean(act_1, axis=0)
    mu_2 = tf.reduce_mean(act_2, axis=0)

    c_1 = tfp.stats.covariance(act_1)
    c_2 = tfp.stats.covariance(act_2)

    sqrt_c_1_mult_c_2 = tf.linalg.sqrtm(tf.matmul(c_1, c_2))

    # Handle nan, complex numbers and infinity
    if tf.math.reduce_any(tf.math.is_nan(sqrt_c_1_mult_c_2)) or tf.math.reduce_any(tf.math.is_inf(sqrt_c_1_mult_c_2)):
        sqrt_c_1_mult_c_2 = tf.cast(tf.linalg.sqrtm(tf.cast(c_1 @ c_2, tf.complex64)), tf.float32)

    if tf.math.reduce_any(tf.math.is_complex(sqrt_c_1_mult_c_2)):
        sqrt_c_1_mult_c_2 = tf.math.real(sqrt_c_1_mult_c_2)

    fid = tf.reduce_sum(tf.square(mu_1 - mu_2)) + tf.linalg.trace(c_1 + c_2 - 2.0 * sqrt_c_1_mult_c_2)
    return fid.numpy().item()

def save_images(images, save_dir, prefix="img"):
    os.makedirs(save_dir, exist_ok=True)
    for i, img in enumerate(images):
        if isinstance(img, tf.Tensor):
            img = img.numpy()
        if len(img.shape) == 4:
            img = np.squeeze(img)
        if img.dtype == np.float32:
            if img.min() < 0 or img.max() > 1:
                img = (img + 1) * 127.5
            else:
                img = img * 255
            img = img.astype(np.uint8)
        
        if len(img.shape) == 2:
            img = np.expand_dims(img, axis=-1)
            img = np.repeat(img, 3, axis=-1)

        img_pil = Image.fromarray(img)
        img_pil.save(os.path.join(save_dir, f"{prefix}_{i}.png"))

if __name__ == '__main__':
    main()