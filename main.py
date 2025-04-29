# train only: python main.py --dataset <dataset> --epochs <epochs> --train --save_imgs
# resume training: python main.py --dataset <dataset> --epochs <epochs> --resume --load_epoch <load_epoch> --train --save_imgs
# resume training: python main.py --dataset <dataset> --epochs <epochs> --resume --load_current --train --save_imgs
# test only: python main.py --dataset <dataset> --resume --load_best_epoch --test
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
import pickle
import os

import matplotlib.pyplot as plt

inception_model = InceptionV3(include_top=False, weights='imagenet', pooling='avg')
clip_tokenizer = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32", use_fast=False, do_rescale=False)
clip_model = TFCLIPModel.from_pretrained("openai/clip-vit-base-patch32")

def parse_args():
    parser = argparse.ArgumentParser(description="Train and Test ControlNet")
    parser.add_argument('--dataset', type=str, default='facesynthetics', choices=['fill50k', 'facesynthetics'])
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--save_dir', type=str, default='./checkpoints', help='Directory to save checkpoints and images')
    parser.add_argument('--img_size', type=int, default=256)
    parser.add_argument('--resume', action='store_true') # Set to True to resume training
    parser.add_argument('--load_dir', type=str, default='./checkpoints')
    parser.add_argument('--load_epoch', type=int, default=0, help='Epoch to load weights from')
    parser.add_argument('--load_best', action='store_true', help='Load best weights') # Set to True to load best weights
    parser.add_argument('--load_current', action='store_true', help='Load current weights') # Set to True to load current weights
    parser.add_argument('--load_best_epoch', type=int, default=0, help='Epoch to load best weights from')
    parser.add_argument('--train', action='store_true', help='Train the model') # Set to True to train the model
    parser.add_argument('--save_imgs', action='store_true', help='Save images') # Set to True to save images
    parser.add_argument('--test', action='store_true', help='Test the model') # Set to True to test the model
    return parser.parse_args()

def main():
    args = parse_args()

    os.makedirs(args.save_dir, exist_ok=True) # if exists, do nothing

    model = ControlSDB(optimizer=tf.keras.optimizers.Adam(learning_rate=args.lr), img_height=args.img_size, img_width=args.img_size)

    if args.dataset == 'fill50k':
        from test_imgs import fill50k
        train_dataset, test_dataset, dataset_length = fill50k.get_dataset(model, args.batch_size, args.img_size)
    else:
        from test_imgs import facesynthetics
        train_dataset, test_dataset, dataset_length = facesynthetics.get_dataset(model, batch_size=args.batch_size, img_size=args.img_size)

    # build model
    dummy_input_shape = (args.batch_size, args.img_size, args.img_size, 3)
    model.build(dummy_input_shape)

    # load weights if available
    continue_in_batches = False
    start_epoch = None
    best_epoch_loss = None
    epoch_losses = None
    batch_losses = None
    if args.resume and args.load_current:
        print("Loading current weights...")
        try:
            model.control_model.load_weights(f"{args.load_dir}/controlnet_current.weights.h5")
            model.diffuser.load_weights(f"{args.load_dir}/unet_current.weights.h5")
            with open(f"{args.load_dir}/info_current.pkl", "rb") as f:
                info = pickle.load(f)
                if len(info) == 6:
                    start_epoch, start_batch_num, epoch_loss, best_epoch_loss, best_batch_loss, epoch_losses = info
                else:
                    start_epoch, start_batch_num, epoch_loss, best_epoch_loss, best_batch_loss, epoch_losses, batch_losses = info
                continue_in_batches = True
            print(f"Loaded current weights successfully.")
        except Exception as e:
            print("Failed to load weights:", e)
    elif args.resume and args.load_best:
        print("Loading best weights...")
        try:
            model.control_model.load_weights(f"{args.load_dir}/controlnet_best.weights.h5")
            model.diffuser.load_weights(f"{args.load_dir}/unet_best.weights.h5")
            with open(f"{args.load_dir}/info_best.pkl", "rb") as f:
                info = pickle.load(f)
                if len(info) == 6:
                    start_epoch, start_batch_num, epoch_loss, best_epoch_loss, best_batch_loss, epoch_losses = info
                else:
                    start_epoch, start_batch_num, epoch_loss, best_epoch_loss, best_batch_loss, epoch_losses, batch_losses = info
                continue_in_batches = True
            print("Loaded best weights successfully.")
        except Exception as e:
            print("Failed to load best weights:", e)
    elif args.resume and args.load_best_epoch:
        print("Loading best epoch weights...")
        try:
            model.control_model.load_weights(f"{args.load_dir}/controlnet_best_epoch.weights.h5")
            model.diffuser.load_weights(f"{args.load_dir}/unet_best_epoch.weights.h5")
            with open(f"{args.load_dir}/info_best_epoch.pkl", "rb") as f:
                info = pickle.load(f)
                if len(info) == 6:
                    start_epoch, start_batch_num, epoch_loss, best_epoch_loss, best_batch_loss, epoch_losses = info
                else:
                    start_epoch, start_batch_num, epoch_loss, best_epoch_loss, best_batch_loss, epoch_losses, batch_losses = info
            print("Loaded best epoch weights successfully.")
        except Exception as e:
            print("Failed to load best epoch weights:", e)
    elif args.resume and args.load_epoch>0:
        print("Loading epoch weights...")
        try:
            model.control_model.load_weights(f"{args.load_dir}/controlnet_epoch_{args.load_epoch}.weights.h5")
            model.diffuser.load_weights(f"{args.load_dir}/unet_epoch_{args.load_epoch}.weights.h5")
            with open(f"{args.load_dir}/info_{args.load_epoch}.pkl", "rb") as f:
                info = pickle.load(f)
                if len(info) == 6:
                    start_epoch, start_batch_num, epoch_loss, best_epoch_loss, best_batch_loss, epoch_losses = info
                else:
                    start_epoch, start_batch_num, epoch_loss, best_epoch_loss, best_batch_loss, epoch_losses, batch_losses = info
            print("Loaded epoch weights successfully.")
        except Exception as e:
            print("Failed to load epoch weights:", e)
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

    if args.train:
        print("----------Start Training----------")

        if epoch_losses is None:
            epoch_losses = []
        if batch_losses is None:
            batch_losses = []

        # Uncomment the following lines to test training the model with a single batch
        # for epoch in range(args.epochs):
        #     for batch in train_dataset.take(1):
        #         losses.append(model.train_step(batch)['loss'])
        #         print(f"Epoch {epoch+1}/{args.epochs}, Batch Loss: {losses[-1]:.6f}")

        if start_epoch is None:
            start_epoch = 0
        if best_epoch_loss is None:
            best_epoch_loss = 0.01
        for epoch in range(start_epoch, start_epoch + args.epochs):
            if not continue_in_batches:
                epoch_loss = 0
                best_batch_loss = 0.01
                start_batch_num = 0
            continue_in_batches = False

            batch_num = 0
            for batch in train_dataset:
                if batch_num < start_batch_num:
                    batch_num += 1
                    continue

                loss = model.train_step(batch)
                batch_losses.append(loss['loss'])
                epoch_loss += loss['loss']
                batch_num += 1
                print(f"Epoch {epoch+1}/{args.epochs}, Batch {batch_num}/{dataset_length // args.batch_size} Loss: {loss['loss']:.6f}")
                
                # save the model weights after every 100 batches
                if batch_num % 20 == 0:
                    try:
                        model.control_model.save_weights(f"{args.save_dir}/controlnet_current.weights.h5")
                        model.diffuser.save_weights(f"{args.save_dir}/unet_current.weights.h5")
                        with open(f"{args.save_dir}/info_current.pkl", "wb") as f:
                            pickle.dump([epoch, batch_num, epoch_loss, best_epoch_loss, best_batch_loss, epoch_losses, batch_losses], f)
                        print(f"Saved current weights for epoch {epoch+1}, batch {batch_num}.")
                    except Exception as e:
                        print("Failed to save weights:", e)

                if loss['loss'] < best_batch_loss:
                    try:
                        model.control_model.save_weights(f"{args.save_dir}/controlnet_best.weights.h5")
                        model.diffuser.save_weights(f"{args.save_dir}/unet_best.weights.h5")
                        with open(f"{args.save_dir}/info_best.pkl", "wb") as f:
                            pickle.dump([epoch, batch_num, epoch_loss, best_epoch_loss, best_batch_loss, epoch_losses, batch_losses], f)
                        print(f"Saved best weights for epoch {epoch+1}, batch {batch_num}.")
                    except Exception as e:
                        print("Failed to save weights:", e)

            avg_epoch_loss = epoch_loss / len(train_dataset)
            epoch_losses.append(avg_epoch_loss)
            print(f"Epoch {epoch+1}/{args.epochs}, Loss: {avg_epoch_loss:.6f}")
            
            # Save the model weights after each epoch
            try:
                model.control_model.save_weights(f"{args.save_dir}/controlnet_epoch_{epoch+1}.weights.h5")
                model.diffuser.save_weights(f"{args.save_dir}/unet_epoch_{epoch+1}.weights.h5")
                with open(f"{args.save_dir}/info_{epoch+1}.pkl", "wb") as f:
                    pickle.dump([epoch + 1, 0, 0, 0, 0, epoch_losses, batch_losses], f)
                print(f"Saved weights for epoch {epoch+1}.")
            except Exception as e:
                print("Failed to save weights:", e)

            if avg_epoch_loss < best_epoch_loss:
                best_epoch_loss = avg_epoch_loss
                try:
                    model.control_model.save_weights(f"{args.save_dir}/controlnet_best_epoch.weights.h5")
                    model.diffuser.save_weights(f"{args.save_dir}/unet_best_epoch.weights.h5")
                    with open(f"{args.save_dir}/info_best_epoch.pkl", "wb") as f:
                        pickle.dump([epoch + 1, 0, 0, 0, 0, epoch_losses, batch_losses], f)
                    print(f"Saved best weights for epoch {epoch+1}.")
                except Exception as e:
                    print("Failed to save weights:", e)
        
        print("----------Finish Training----------")

    if args.save_imgs:
        epochs = list(range(1, len(epoch_losses) + 1))
        plt.plot(epochs, epoch_losses, label='Loss')
        plt.title('Training Loss over Epochs')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.savefig(f"{args.save_dir}/loss_curve.png")
        plt.clf()
        plt.cla()
        plt.close('all')

    if args.test:
        print("----------Start Testing----------")
        captions = [data['txt'] for data in test_dataset]
        captions = []
        for batch in test_dataset:
            captions.extend(batch['txt'].numpy().tolist())
        captions = [c.decode('utf-8') if isinstance(c, bytes) else c for c in captions]
        generated_images_sd = []
        stable_diffusion = keras_cv.models.StableDiffusion(img_width=args.img_size, img_height=args.img_size)

        for caption in captions[0:args.batch_size]:
            generated_images_sd.append(stable_diffusion.text_to_image(caption, batch_size=1)[0])
        
        save_image_pil(np.array(generated_images_sd), "outputs/sd", "sd")

        generated_images_controlnet_clip = []
        generated_images_controlnet_fid = []

        batch_num = 0
        for batch in test_dataset:
            images = model.predict(batch)
            print(tf.shape(images))
            generated_images_controlnet_clip.extend(images.numpy())
            generated_images_controlnet_fid.append(images.numpy())
            print(images.numpy())
            save_image_pil(images.numpy(), "outputs/cn", "cn", batch_num)
            batch_num += 1

        clip_score_sd = calculate_clip_score(generated_images_sd, captions[0:args.batch_size])
        clip_score_controlnet = calculate_clip_score(generated_images_controlnet_clip, captions[0:args.batch_size])
        print("CLIP Score:")
        print("Stable Diffusion: " + str(clip_score_sd))
        print("Control Net: " + str(clip_score_controlnet))

        real_images = [data['jpg'] for data in test_dataset]
        fid_score_sd = calculate_fid_score(real_images[0:args.batch_size], tf.convert_to_tensor(generated_images_sd))
        fid_score_controlnet = calculate_fid_score(real_images[0:args.batch_size], generated_images_controlnet_fid)
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
    
    clip_score = []

    for i, image in enumerate(images):
        if image.dtype != np.uint8:
            image = np.clip(image, 0, 255).astype(np.uint8) / 255.0

        inputs = clip_tokenizer(
            text=captions[i],
            images=image,
            return_tensors="tf",
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

        clip_score.append(tf.reduce_mean(similarity).numpy().item())
    
    return np.mean(clip_score)
    
def get_activations(images: tf.Tensor):
    # Assuming InceptionV3 accepts 299x299 images
    images_resized = tf.image.resize(images, (299, 299))

    images_resized = preprocess_input(images_resized)

    activations = inception_model(images_resized, training=False)
    return activations

def calculate_fid_score(real_images, generated_images):
    real_images = tf.concat(real_images, axis=0)
    generated_images = tf.concat(generated_images, axis=0)

    # d^2 = ||mu_1 – mu_2||^2 + Tr(c_1 + c_2 – 2*sqrt(c_1*c_2))
    act_1 = get_activations(real_images)
    act_2 = get_activations(generated_images)

    mu_1 = tf.reduce_mean(act_1, axis=0)
    mu_2 = tf.reduce_mean(act_2, axis=0)

    c_1 = tfp.stats.covariance(act_1)
    c_2 = tfp.stats.covariance(act_2)

    try:
        sqrt_c_1_mult_c_2 = tf.linalg.sqrtm(tf.matmul(c_1, c_2))
        if tf.math.reduce_any(tf.math.is_nan(sqrt_c_1_mult_c_2)):
            raise ValueError("NaN in sqrtm")
        sqrt_c_1_mult_c_2 = tf.math.real(sqrt_c_1_mult_c_2)
    except:
        s, u, v = tf.linalg.svd(tf.matmul(c_1, c_2))
        sqrt_c_1_mult_c_2 = u @ tf.linalg.diag(tf.sqrt(s)) @ tf.transpose(v)

    if tf.math.reduce_any(tf.math.is_nan(sqrt_c_1_mult_c_2)) or tf.math.reduce_any(tf.math.is_inf(sqrt_c_1_mult_c_2)):
        sqrt_c_1_mult_c_2 = tf.cast(tf.linalg.sqrtm(tf.cast(c_1 @ c_2, tf.complex64)), tf.float32)

    if tf.math.reduce_any(tf.math.imag(sqrt_c_1_mult_c_2) != 0):
        sqrt_c_1_mult_c_2 = tf.math.real(sqrt_c_1_mult_c_2)

    fid = tf.reduce_sum(tf.square(mu_1 - mu_2)) + tf.linalg.trace(c_1 + c_2 - 2.0 * sqrt_c_1_mult_c_2)
    return fid.numpy().item()

def save_image_pil(image_batch, output_dir, prefix="img", batch_num=0):
    os.makedirs(output_dir, exist_ok=True)

    for i in range(image_batch.shape[0]):
        img = image_batch[i]
        if img.dtype != np.uint8:
            if img.min() < 0 or img.max() > 255:
                img = (img + 1) * 127.5
            img = img.astype(np.uint8)
    
        Image.fromarray(img).save(os.path.join(output_dir, f"{prefix}_{batch_num + i}.png"))

if __name__ == '__main__':
    main()