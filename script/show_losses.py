import argparse
import matplotlib.pyplot as plt
import pickle

def parse_args():
    parser = argparse.ArgumentParser(description="Show plot of losses")
    parser.add_argument('--file_path', type=str, default='./checkpoints/info_current.pkl')
    parser.add_argument('--save_dir', type=str, default='./checkpoints', help='Directory to save images')
    return parser.parse_args()

def main():
    args = parse_args()

    with open(args.file_path, "rb") as f:
        _, _, _, _, _, epoch_losses, batch_losses = pickle.load(f)

    epochs = list(range(1, len(epoch_losses) + 1))
    plt.plot(epochs, epoch_losses, label='Loss')
    plt.title('Training Loss over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.savefig(f"{args.save_dir}/epoch_loss.png")
    plt.clf()
    plt.cla()

    batches = list(range(1, len(batch_losses) + 1))
    plt.plot(batches, batch_losses, label='Loss')
    plt.title('Training Loss over Batches')
    plt.xlabel('Batches')
    plt.ylabel('Loss')
    plt.savefig(f"{args.save_dir}/batch_loss.png")
    plt.clf()
    plt.cla()
    plt.close('all')

if __name__ == '__main__':
    main()