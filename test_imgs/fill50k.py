# 1. download fill50k.zip from https://huggingface.co/lllyasviel/ControlNet/tree/main/training
# 2. make a 'training' folder in the root directory of this repo
# 3. unzip and move 'fill50k' folder to 'training' folder
# training/fill50k/source
# training/fill50k/target
# training/fill50k/prompt.json

import json
import cv2
import numpy as np
import tensorflow as tf

def load_data(img_size):
    with open('./training/fill50k/prompt.json', 'rt') as f:
        for line in f:
            item = json.loads(line)

            source_filename = item['source']
            target_filename = item['target']
            prompt = item['prompt']

            source = cv2.imread('./training/fill50k/' + source_filename)
            target = cv2.imread('./training/fill50k/' + target_filename)

            # BRG to RGB
            source = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
            target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)

            # normalize source to 0-1 range
            source = source.astype(np.float32) / 255.0

            # normalize target to -1 to 1 range
            target = (target.astype(np.float32) / 127.5) - 1.0

            # resize to img_size x img_size
            source = cv2.resize(source, (img_size, img_size))
            target = cv2.resize(target, (img_size, img_size))

            yield {
                'jpg': target,
                'hint': source,
                'txt': prompt
            }

def get_dataset(batch_size, img_size):
    dataset = tf.data.Dataset.from_generator(
        lambda: load_data(img_size),
        output_signature={
            'jpg': tf.TensorSpec(shape=(None, None, 3), dtype=tf.float32),
            'hint': tf.TensorSpec(shape=(None, None, 3), dtype=tf.float32),
            'txt': tf.TensorSpec(shape=(), dtype=tf.string)
        }
    )

    dataset_length = 50000

    # split dataset into train and test
    train_size = int(dataset_length * 0.8)
    train_dataset = dataset.take(train_size)
    test_dataset = dataset.skip(train_size)

    train_dataset = train_dataset.shuffle(1000).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    test_dataset = test_dataset.batch(batch_size)

    return train_dataset, test_dataset, dataset_length