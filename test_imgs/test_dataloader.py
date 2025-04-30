# run to test the dataloader: 
# python test_imgs/test_dataloader.py --fill50k
# python test_imgs/test_dataloader.py --facesynthetics

import json
import numpy as np
import tensorflow as tf
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
import fill50k
import facesynthetics

# import ControlSDB from ../cldm/cldm.py
from cldm.cldm import ControlSDB

import matplotlib.pyplot as plt 

def test_get_dataset():
    # parse command line arguments
    parser = argparse.ArgumentParser(description="Test data loader")
    parser.add_argument('--fill50k', action='store_true', help='Use fill50k dataset')
    parser.add_argument('--facesynthetics', action='store_true', help='Use facesynthetics dataset')
    args = parser.parse_args()

    model = ControlSDB(optimizer=tf.keras.optimizers.Adam(), img_height=256, img_width=256)

    # select dataset based on command line argument
    if args.fill50k:
        print("Using fill50k dataset")
        get_dataset = fill50k.get_dataset
    elif args.facesynthetics:
        print("Using facesynthetics dataset")
        get_dataset = facesynthetics.get_dataset
    else:
        print("Please specify a dataset with --fill50k or --facesynthetics")
        return

    train_dataset, test_dataset, dataset_length = get_dataset(model, batch_size=1, img_size=256)

    for i, data in enumerate(train_dataset):
        print(f"Batch {i}:")
        print(f"jpg shape: {data['jpg'].shape}")
        print(f"hint shape: {data['hint'].shape}")

        # plot the image example
        plt.imshow(data['jpg'][0])
        plt.axis('off')
        plt.title("jpg")
        plt.show()

        plt.imshow(data['hint'][0])
        plt.axis('off')
        plt.title("hint")
        plt.show()
        
        print(f"txt: {data['txt']}")
        if i == 1:
            break
    print("Test for train dataset completed.")
    
    # for i, data in enumerate(test_dataset):
    #     print(f"Batch {i}:")
    #     print(f"jpg shape: {data['jpg'].shape}")
    #     print(f"hint shape: {data['hint'].shape}")
    #     print(f"txt: {data['txt']}")
    #     if i == 1:
    #         break
    # print("Test for test dataset completed.")
    
if __name__ == "__main__":
    test_get_dataset()
