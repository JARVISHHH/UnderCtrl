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

def test_get_dataset():
    # parse command line arguments
    parser = argparse.ArgumentParser(description="Test data loader")
    parser.add_argument('--fill50k', action='store_true', help='Use fill50k dataset')
    parser.add_argument('--facesynthetics', action='store_true', help='Use facesynthetics dataset')
    args = parser.parse_args()

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

    dataset = get_dataset()

    for i, data in enumerate(dataset):
        print(f"Batch {i}:")
        print(f"jpg shape: {data['jpg'].shape}")
        print(f"hint shape: {data['hint'].shape}")
        print(f"txt: {data['txt']}")
        if i == 1:
            break
    print("Test completed.")

if __name__ == "__main__":
    test_get_dataset()
