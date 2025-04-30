import tensorflow as tf
from datasets import load_dataset
import numpy as np
from PIL import Image  # Ensure correct image handling
import os

def preprocess_example(example, img_size, model):
    target = example["image"].convert("RGB").resize((img_size, img_size))
    source = example["spiga_seg"].convert("RGB").resize((img_size, img_size))

    target = np.array(target).astype(np.float32)
    source = np.array(source).astype(np.float32)

    source = source / 255.0
    target = (target / 127.5) - 1.0

    caption = example["image_caption"]
    # caption = caption.decode("utf-8") if isinstance(caption, bytes) else caption
    encoded_text = model.encode_text(caption)  # (1, 77, 768)
    encoded_text = np.squeeze(encoded_text, axis=0)  # (77, 768)

    return {
        "jpg": target,
        "hint": source,
        "txt": encoded_text,
        "str": caption
    }

def make_generator(dataset, img_size, model):
    def gen():
        for example in dataset:
            yield preprocess_example(example, img_size, model)
    return gen

def get_dataset(model, batch_size=8, img_size=256, test_size=0.2):
    # Load and split dataset
    
    # if os.path.exists("training/face_hf_dataset_split"):
    #     hf_dataset_split = load_dataset("training/face_hf_dataset_split")
    #     print("Loaded dataset from disk.")
    # else:

    os.makedirs("training", exist_ok=True)
    hf_dataset = load_dataset("multimodalart/facesyntheticsspigacaptioned", split="train")
    hf_dataset_split = hf_dataset.train_test_split(test_size=test_size, seed=42)
    # hf_dataset_split.save_to_disk("training/face_hf_dataset_split")
    # print("Dataset split and saved to disk.")

    train_set = hf_dataset_split["train"]
    test_set = hf_dataset_split["test"]    

    output_signature = {
        "jpg": tf.TensorSpec(shape=(img_size, img_size, 3), dtype=tf.float32),
        "hint": tf.TensorSpec(shape=(img_size, img_size, 3), dtype=tf.float32),
        "txt": tf.TensorSpec(shape=(77, 768), dtype=tf.float32),
        "str": tf.TensorSpec(shape=(), dtype=tf.string)
    }

    train_data = tf.data.Dataset.from_generator(
        make_generator(train_set, img_size, model),
        output_signature=output_signature
    )

    test_data = tf.data.Dataset.from_generator(
        make_generator(test_set, img_size, model),
        output_signature=output_signature
    )

    train_data = train_data.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    test_data = test_data.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    return train_data, test_data, len(train_set) + len(test_set)

# Usage example:
# train_ds, test_ds, train_len, test_len = get_dataset(your_model)
