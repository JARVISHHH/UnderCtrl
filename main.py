import argparse
import tensorflow as tf
from cldm.cldm import ControlSDB

def parse_args():
    parser = argparse.ArgumentParser(description="Train and Test ControlNet")
    parser.add_argument('--mode', choices=['train', 'test'], required=True, help='Mode: train or test')
    parser.add_argument('--dataset', type=str, default='fill50k', choices=['fill50k', 'facesynthetics'])
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--save_dir', type=str, default='./checkpoints')
    return parser.parse_args()

def main():
    args = parse_args()

    if args.dataset == 'fill50k':
        from test_imgs import fill50k
        dataset = fill50k.get_dataset()
        dataset_length = 50000
    else:
        from test_imgs import facesynthetics
        dataset = facesynthetics.get_dataset()
        dataset_length = 50000  # TODO

    # split dataset into train and test
    train_size = int(dataset_length * 0.8)
    train_dataset = dataset.take(train_size)
    test_dataset = dataset.skip(train_size)

    model = ControlSDB()

    optimizer = tf.keras.optimizers.Adam(learning_rate=args.lr)
    model.compile(
        optimizer=optimizer,
        loss="mse",
    )
    model.fit(train_dataset, epochs=args.epochs)
    
if __name__ == '__main__':
    main()