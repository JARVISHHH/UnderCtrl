# train only: python main.py --dataset <dataset> --epochs <epochs> --train --save_imgs
# resume training: python main.py --dataset <dataset> --epochs <epochs> --resume --load_epoch <load_epoch> --train --save_imgs
# resume training: python main.py --dataset <dataset> --epochs <epochs> --resume --load_current --train --save_imgs
# test only: python main.py --dataset <dataset> --resume --load_best_epoch --test
# keep batch size same for train and test
# inference: python main.py --dataset <dataset> --resume --load_current --inference

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
from cleanfid import fid

import matplotlib.pyplot as plt

inception_model = InceptionV3(include_top=False, weights='imagenet', pooling='avg')
clip_tokenizer = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32", use_fast=False)
clip_model = TFCLIPModel.from_pretrained("openai/clip-vit-base-patch32")

def parse_args():
    parser = argparse.ArgumentParser(description="Train and Test ControlNet")
    parser.add_argument('--dataset', type=str, default='facesynthetics', choices=['fill50k', 'facesynthetics'])
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--save_dir', type=str, default='./checkpoints', help='Directory to save checkpoints and images')
    parser.add_argument('--img_size', type=int, default=256)
    parser.add_argument('--test_size', type=float, default=0.2)
    parser.add_argument('--resume', action='store_true') # Set to True to resume training
    parser.add_argument('--load_dir', type=str, default='./checkpoints')
    parser.add_argument('--load_epoch', type=int, default=0, help='Epoch to load weights from')
    parser.add_argument('--load_best', action='store_true', help='Load best weights') # Set to True to load best weights
    parser.add_argument('--load_current', action='store_true', help='Load current weights') # Set to True to load current weights
    parser.add_argument('--load_best_epoch', type=int, default=0, help='Epoch to load best weights from')
    parser.add_argument('--train', action='store_true', help='Train the model') # Set to True to train the model
    parser.add_argument('--save_imgs', action='store_true', help='Save images') # Set to True to save images
    parser.add_argument('--test', action='store_true', help='Test the model') # Set to True to test the model
    parser.add_argument('--inference', action='store_true', help='Run inference') # Set to True to run inference
    return parser.parse_args()

def main():
    args = parse_args()

    os.makedirs(args.save_dir, exist_ok=True) # if exists, do nothing
    os.makedirs("outputs", exist_ok=True) # if exists, do nothing

    model = ControlSDB(optimizer=tf.keras.optimizers.Adam(learning_rate=args.lr), img_height=args.img_size, img_width=args.img_size)
    stable_diffusion = keras_cv.models.StableDiffusion(img_width=args.img_size, img_height=args.img_size)

    if args.dataset == 'fill50k':
        from test_imgs import fill50k
        train_dataset, test_dataset, dataset_length = fill50k.get_dataset(model, args.batch_size, args.img_size, test_size=args.test_size)
        
    else:
        from test_imgs import facesynthetics
        train_dataset, test_dataset, dataset_length = facesynthetics.get_dataset(model, batch_size=args.batch_size, img_size=args.img_size, test_size=args.test_size)
    train_dataset_length = (int)(dataset_length * (1 - args.test_size))
    test_dataset_length = dataset_length - train_dataset_length

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
            print(f"Loaded current weights successfully.")
        except Exception as e:
            print("Failed to load weights:", e)

        try:
            with open(f"{args.load_dir}/info_current.pkl", "rb") as f:
                info = pickle.load(f)
                if len(info) == 6:
                    start_epoch, start_batch_num, epoch_loss, best_epoch_loss, best_batch_loss, epoch_losses = info
                else:
                    start_epoch, start_batch_num, epoch_loss, best_epoch_loss, best_batch_loss, epoch_losses, batch_losses = info
                continue_in_batches = True
            print("Loaded current weights info successfully.")
        except Exception as e:
            print("Failed to load current weights info:", e)
            continue_in_batches = False
            start_epoch = None
            best_epoch_loss = None
            epoch_losses = None
            batch_losses = None
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
            # file = keras.utils.get_file(
            #                         origin="https://huggingface.co/fchollet/stable-diffusion/resolve/main/kcv_diffusion_model.h5",  # noqa: E501
            #                         file_hash="8799ff9763de13d7f30a683d653018e114ed24a6a819667da4f5ee10f9e805fe",  # noqa: E501
            # )
            # print("Loading control model.")
            # original_weights = {layer.name: layer.get_weights() for layer in model.control_model.layers}
            # model.control_model.load_weights(file, skip_mismatch=True)
            # for layer in model.control_model.layers:
            #     before = original_weights[layer.name]
            #     after = layer.get_weights()
            #     if any((b != a).any() for b, a in zip(before, after)):
            #         print(f"Control Model weights loaded for layer: {layer.name}")

            # print("Loading diffuser model.")
            # original_weights = {layer.name: layer.get_weights() for layer in model.diffuser.layers}
            # model.diffuser.load_weights(file)
            # for layer in model.diffuser.layers:
            #     before = original_weights[layer.name]
            #     after = layer.get_weights()
            #     if any((b != a).any() for b, a in zip(before, after)):
            #         print(f"Diffuser weights loaded for layer: {layer.name}")

            keras.backend.clear_session()

            original_diffusion_model = keras_cv.models.stable_diffusion.DiffusionModel(args.img_size, args.img_size, 77)

            original_weights = {layer.name: layer.get_weights() for layer in original_diffusion_model.layers}

            for layer in model.control_model.layers:
                if layer.name in original_weights:
                    try:
                        layer.set_weights(original_weights[layer.name])
                        print(f"[Loaded] {layer.name}")
                    except Exception as e:
                        print(f"[Mismatch] {layer.name}: {e}")
                else:
                    print(f"[Skipped] {layer.name} (not found)")

            for layer in model.diffuser.layers:
                if layer.name in original_weights:
                    try:
                        layer.set_weights(original_weights[layer.name])
                        print(f"[Loaded] {layer.name}")
                    except Exception as e:
                        print(f"[Mismatch] {layer.name}: {e}")
                else:
                    print(f"[Skipped] {layer.name} (not found)")

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
                print(f"Epoch {epoch+1}/{args.epochs}, Batch {batch_num}/{(train_dataset_length - 1) // args.batch_size + 1} Loss: {loss['loss']:.6f}")
                
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

            avg_epoch_loss = epoch_loss / batch_num
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

    if args.inference:
        print("----------Start Inference----------")

        # for layer in model.diffuser.layers:
        #     before = original_weights[layer.name]
        #     after = layer.get_weights()
        #     if any((b != a).any() for b, a in zip(before, after)):
        #         print(f"Diffuser weights different for layer: {layer.name}")
                
        # print("compare finished")

        iterator = iter(test_dataset)
        sample = next(iterator)

        # (8, 256, 256, 3) => (1, 256, 256, 3)
        sample = {k: v[0] for k, v in sample.items()}
        sample = {k: tf.expand_dims(v, axis=0) for k, v in sample.items()}

        text = sample['str']
        text_str = text.numpy()[0].decode('utf-8')
        image = sample['jpg']
        hint = sample['hint']

        # generate cn image
        cn_image = model.predict(sample)

        # generate sd image
        sd_image = stable_diffusion.text_to_image(text_str, batch_size=1)[0]

        # plot image, hint, sd_image, cn_image, text as title
        fig, axs = plt.subplots(2, 2, figsize=(10, 10))
        axs[0, 0].imshow((image[0] + 1) / 2)
        axs[0, 0].set_title("Original Image")
        axs[0, 1].imshow(hint[0])
        axs[0, 1].set_title("Hint Image")
        axs[1, 0].imshow(sd_image)
        axs[1, 0].set_title("Stable Diffusion Image")
        axs[1, 1].imshow(cn_image[0])
        axs[1, 1].set_title("ControlNet Image")
        for ax in axs.flat:
            ax.axis('off')
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.suptitle(text_str, fontsize=16)
        plt.savefig(f"outputs/inference_{text_str}.png")
        print(f"Saved inference image as outputs/inference_{text_str}.png")
        # plt.show()

        save_image_pil(np.array([sd_image]), "outputs/sd", "sd")
        save_image_pil(cn_image, "outputs/cn", "cn")
        save_image_pil(np.array([image[0]]), "outputs/jpg", "jpg")
        
        clip_sd = calculate_clip_score([sd_image], [text_str])
        clip_cn = calculate_clip_score([cn_image], [text_str])
        print('CLIP Score:')
        print("SD:", clip_sd)
        print("CN:", clip_cn)

        fid_sd = calculate_fid_score('outputs/jpg', 'outputs/sd')
        fid_cn = calculate_fid_score('outputs/jpg', 'outputs/cn')
        print('FID Score:')
        print("SD:", fid_sd)
        print("CN:", fid_cn)

        print("----------Finish Inference----------")

def calculate_clip_score(image, text):
    inputs = clip_tokenizer(text=text, images=image, return_tensors="tf", padding=True)
    outputs = clip_model(**inputs)
    image_embeds = outputs.image_embeds 
    text_embeds = outputs.text_embeds
    image_embeds = tf.math.l2_normalize(image_embeds, axis=-1)
    text_embeds = tf.math.l2_normalize(text_embeds, axis=-1)

    return tf.reduce_sum(image_embeds * text_embeds, axis=-1)

def calculate_fid_score(real_dir, generated_dir):
    return fid.compute_fid(real_dir, generated_dir, device='cpu', num_workers=0)

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