import argparse
import tensorflow as tf
from cldm.cldm import ControlSDB
import keras_cv
import numpy as np
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
import tensorflow_probability as tfp
from transformers import TFCLIPModel, CLIPProcessor
from PIL import Image
import os

import matplotlib.pyplot as plt

inception_model = InceptionV3(include_top=False, weights='imagenet', pooling='avg')
clip_tokenizer = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
clip_model = TFCLIPModel.from_pretrained("openai/clip-vit-base-patch32")

def parse_args():
    parser = argparse.ArgumentParser(description="Train and Test ControlNet")
    parser.add_argument('--dataset', type=str, default='fill50k', choices=['fill50k', 'facesynthetics'])
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--save_dir', type=str, default='./checkpoints')
    parser.add_argument('--img_size', type=int, default=64)
    return parser.parse_args()

def main():
    args = parse_args()

    if args.dataset == 'fill50k':
        from test_imgs import fill50k
        dataset = fill50k.get_dataset(args.batch_size, args.img_size)
        dataset_length = 50000
    else:
        from test_imgs import facesynthetics
        dataset = facesynthetics.get_dataset(batch_size=args.batch_size, img_size=args.img_size)
        dataset_length = 50000  # TODO

    # split dataset into train and test
    train_size = int(dataset_length * 0.08)
    train_dataset = dataset.take(train_size)
    test_dataset = dataset.skip(train_size)

    model = ControlSDB(optimizer=tf.keras.optimizers.Adam(learning_rate=args.lr), img_height=args.img_size, img_width=args.img_size)

    print("----------Start Training----------")

    losses = []

    for _ in range(args.epochs):
        for batch in train_dataset.take(1):
            losses.append(model.train_step(batch)['loss'])
    
    print("----------Finish Training----------")

    # epochs = list(range(1, len(losses) + 1))
    # plt.plot(epochs, losses, label='Loss')
    # plt.title('Training Loss over Epochs')
    # plt.xlabel('Epochs')
    # plt.ylabel('Loss')
    # plt.legend()
    # plt.show()
    # print('----------Finish graphing-----------')

    print('----------Start Testing----------')
    captions = [data['txt'] for data in test_dataset.take(1)]
    generated_images_sd = []
    stable_diffusion = keras_cv.models.StableDiffusion(img_width=512, img_height=512)
    print('Created stable diffusion')
    for caption in captions:
        if isinstance(caption, tf.Tensor):
            caption_str = caption.numpy()[0]
            if isinstance(caption_str, bytes):
                caption_str = caption_str.decode("utf-8")
        else:
            caption_str = caption
        generated_images_sd.append(stable_diffusion.text_to_image(caption_str, batch_size=args.batch_size))
    generated_images_sd = resize_images(generated_images_sd)
    print('Finish inference in sd')
    generated_images_controlnet = resize_images(model.predict(test_dataset.take(1)))
    print('Finish inference in controlnet')

    save_images(generated_images_sd, save_dir="outputs/sd", prefix="sd")
    save_images(generated_images_controlnet, save_dir="outputs/controlnet", prefix="controlnet")

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

def resize_images(images, target_size=(299, 299)):
    images = tf.convert_to_tensor(images, dtype=tf.float32)
    resized = tf.image.resize(images, size=target_size, method='bilinear')
    return resized

def calculate_clip_score(images, captions):
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
        img_pil = Image.fromarray(img)
        img_pil.save(os.path.join(save_dir, f"{prefix}_{i}.png"))

if __name__ == '__main__':
    main()