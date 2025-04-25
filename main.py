import argparse
import tensorflow as tf 
from cldm import ControlNet, ControlledUnetModel, ControlLDM
import keras_cv
from keras_cv.models.stable_diffusion.diffusion_model import DiffusionModel
from test_imgs import fill50k, facesynthetics

def parse_args():
    parser = argparse.ArgumentParser(description="Train and Test ControlNet")
    parser.add_argument('--mode', choices=['train', 'test'], required=True, help='Mode: train or test')
    parser.add_argument('--dataset', type=str, default='fill50k', choices=['fill50k', 'facesynthetics'])
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--save_dir', type=str, default='./checkpoints')
    return parser.parse_args()

def main():
    args = parse_args()

    if args.dataset == 'fill50k':
        dataset = fill50k.get_dataset()
    else:
        dataset = facesynthetics.get_dataset()

    stable_diffusion = keras_cv.models.StableDiffusion(512, 512)
    control_net = ControlNet()
    controlled_unet = ControlledUnetModel()
    model = ControlLDM(stable_diffusion, control_net, controlled_unet)

    optimizer = tf.keras.optimizers.Adam(learning_rate=args.lr)
    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.MeanAbsoluteError(),
    )
    for _ in range(args.epochs):
        model.train_step(dataset)
    
if __name__ == '__main__':
    main()