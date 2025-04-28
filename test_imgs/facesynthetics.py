import tensorflow as tf
from datasets import load_dataset
import numpy as np

hf_dataset = load_dataset("multimodalart/facesyntheticsspigacaptioned", split='train')

def preprocess_example(example, img_size):
    target = example["image"].convert("RGB").resize((img_size, img_size))
    source = example["spiga_seg"].convert("RGB").resize((img_size, img_size))

    target = np.array(target).astype(np.float32)
    source = np.array(source).astype(np.float32)

    source = source / 255.0
    target = (target / 127.5) - 1.0

    return {
        "jpg": target,
        "hint": source,
        "txt": example["image_caption"]
    }

def generator(img_size):
    for example in hf_dataset:
        yield preprocess_example(example, img_size)


def get_dataset(batch_size=8, img_size=256):
    output_signature = {
        "jpg": tf.TensorSpec(shape=(img_size, img_size, 3), dtype=tf.float32),
        "hint": tf.TensorSpec(shape=(img_size, img_size, 3), dtype=tf.float32),
        "txt": tf.TensorSpec(shape=(), dtype=tf.string),
    }

    dataset = tf.data.Dataset.from_generator(
        lambda: generator(img_size),
        output_signature=output_signature
    )

    dataset_length = len(hf_dataset) # TODO: check if its correct here

    # print(f"Preprocessing {dataset_length} examples...")

    # split dataset into train and test
    train_size = int(dataset_length * 0.8)
    train_data = dataset.take(train_size)
    test_data = dataset.skip(train_size)

    # save the dataset to disk
    train_data = train_data.cache()
    test_data = test_data.cache()

    # print(f"Preprocessing completed.")

    train_data = train_data.shuffle(1000).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    test_data = test_data.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    return train_data, test_data

# Usage example
# dataset = get_dataset()