#!/usr/bin/env python
# Copyright Axelera AI, 2023
#
# Usage - firstly, install tensorflow by `pip install tensorflow==2.14.0`
# To generate TFRecord files:
# python tools/gen_tfrecords.py --dataset_path ./data/ms1m_align_112
#   {--output_path ./data/ms1m_align_112.tfrecord} {--binary_img}
#   {--num_records_per_file 1000} {--class_file ./data/ms1m_align_112.names}
#
# To generate embedding JSON or npy file from the 1st framae of each person:
# python tools/gen_tfrecords.py --dataset_path ./data/ms1m_align_112
#   --onnx_model ./models/model.onnx {--output_path ./data/ms1m_align_112.json}

import json
import logging
from pathlib import Path

from absl import app, flags
import numpy as np
import tensorflow as tf
import tqdm

FLAGS = flags.FLAGS


flags.DEFINE_string('dataset_path', None, 'Path to the dataset directory')
flags.DEFINE_string('output_path', None, 'Output TFRecord file pattern')
flags.DEFINE_integer('num_records_per_file', 1000, 'Number of records per TFRecord file')
flags.DEFINE_boolean('binary_img', False, 'Use binary image encoding')
flags.DEFINE_string(
    'class_file', None, 'Path to class.names or class embedding JSON file if exists'
)
flags.DEFINE_string('onnx_model', None, 'Path to ONNX model to generate embedding')


def transform_image(image):
    '''Update this function to match the preprocessing of the model'''
    image = tf.image.resize(image, [112, 112])
    image = image / 255.0
    return image


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def write_tfrecord_file(writer, img_str, source_id, filename):
    img_str = str.encode(img_str)
    feature = {
        'image/source_id': _int64_feature(source_id),
    }
    if FLAGS.binary_img:
        feature['image/filename'] = _bytes_feature(str.encode(filename))
        feature['image/encoded'] = _bytes_feature(img_str)
    else:
        feature['image/img_path'] = _bytes_feature(img_str)

    tf_example = tf.train.Example(features=tf.train.Features(feature=feature))
    writer.write(tf_example.SerializeToString())


def write_names_file(class_names, output_file):
    with open(output_file, 'w') as names_file:
        names_file.write('\n'.join(class_names))


def generate_embedding(samples, embedding_file, model_path):
    import onnx
    import onnxruntime as rt

    model_path = Path(model_path).expanduser().resolve()
    if not model_path.exists() or not model_path.is_file():
        raise ValueError(f"Provided path '{model_path}' is not a valid file.")

    providers = rt.get_available_providers()
    model = onnx.load(model_path)
    sess = rt.InferenceSession(model.SerializeToString(), providers=providers)
    input_name = sess.get_inputs()[0].name

    # if embedding_file is json, save as json, else save as npy
    if embedding_file.suffix == '.json':
        embedding = {}
        for img_str, id_name, filename in tqdm.tqdm(samples):
            if id_name not in embedding:
                embedding_vector = gen_embedding_per_sample(img_str, sess, input_name)
                embedding[id_name] = embedding_vector.flatten().tolist()
        embedding_file.write_text(json.dumps(embedding, indent=2))
    else:
        embedding = []
        for img_str, id_name, filename in tqdm.tqdm(samples):
            embedding_vector = gen_embedding_per_sample(img_str, sess, input_name)
            embedding.append(embedding_vector.flatten())
        embedding = np.array(embedding)
        np.save(embedding_file, embedding)
    logging.info(f'Embedding saved to {embedding_file}')


def gen_embedding_per_sample(img_file, onnx_model, input_name, transform=transform_image):
    image = tf.io.read_file(img_file)
    image = tf.image.decode_jpeg(image, channels=3)
    image = transform(image)
    image = tf.expand_dims(image, 0)
    embedding = onnx_model.run(None, {input_name: image.numpy()})[0]
    norm = np.linalg.norm(embedding, axis=1, keepdims=True)
    return embedding / norm


def main(_):
    dataset_path = FLAGS.dataset_path
    num_records_per_file = FLAGS.num_records_per_file
    binary_img = FLAGS.binary_img
    class_file = FLAGS.class_file

    if dataset_path is None:
        raise ValueError('The --dataset_path flag is required.')
    elif not Path(dataset_path).is_dir():
        raise ValueError(f'{dataset_path} is not a valid directory.')
    dataset_name = Path(dataset_path).name

    logging.info('Loading {}'.format(dataset_path))
    class_mapping = {}
    class_names = []
    if not class_file:
        class_file = Path(dataset_path).joinpath(f'{dataset_name}.names')
        logging.info(f'Class file not specified. Using {class_file} instead.')
    elif not Path(class_file).is_file():
        raise FileNotFoundError(f'Class file {class_file} not found.')
    else:
        if class_file.name.endswith('.json'):
            with open(class_file, 'r') as json_file:
                class_mapping = json.load(json_file)
            # enumerate names from the class mapping to construct a class mapping and write to a .names file
            # class_mapping is a dict with class names as keys and class vectors as values
            class_names = [class_name for class_name in class_mapping.keys()]
            write_names_file(class_names, class_file.replace('.json', '.names'))
        elif class_file.name.endswith('.names'):
            with open(class_file, 'r') as names_file:
                class_names = names_file.read().splitlines()
                class_mapping = {class_name: index for index, class_name in enumerate(class_names)}

    dataset_path = Path(dataset_path).expanduser().resolve()
    if not dataset_path.exists() or not dataset_path.is_dir():
        raise ValueError(f"Provided path '{dataset_path}' is not a valid directory.")

    # Check if class names are numeric and if not, map them to class labels
    numeric_check_table = np.array([])
    logging.info(
        f"Number of subdirectories in {dataset_path}: {len(list(dataset_path.iterdir()))}"
    )
    for id_name in tqdm.tqdm(dataset_path.iterdir()):
        if not id_name.is_dir():
            continue

        class_name = id_name.name
        try:
            int(class_name)
            numeric_check_table = np.append(numeric_check_table, True)
        except ValueError:
            numeric_check_table = np.append(numeric_check_table, False)

    # all class names should be either numeric or non-numeric
    is_numeric = np.all(numeric_check_table)
    if not (is_numeric ^ np.all(np.logical_not(numeric_check_table))):
        logging.warning('Class names should be all numeric or all non-numeric.')
        is_numeric = False  # treat all class names as non-numeric

    # Enumerate the dataset directory
    img_paths = []
    valid_extensions = {'.bmp', '.jpg', '.jpeg', '.png'}
    with tqdm.tqdm(desc=f"Searching images under {dataset_path}") as pbar:
        for p in dataset_path.rglob('*.*'):
            if p.suffix.lower() in valid_extensions:
                img_paths.append(p)
            pbar.update(1)

    samples = []
    for img_path in img_paths:
        class_name = img_path.parent.name

        if is_numeric:
            source_id = int(class_name)
            class_mapping[class_name] = source_id
        else:
            # If not numeric, map it to a class label
            if class_name not in class_mapping:
                class_mapping[class_name] = len(class_mapping)  # Enumerate new classes
            source_id = class_mapping[class_name]
        if FLAGS.onnx_model:
            samples.append((str(img_path), class_name, img_path.name))
        else:
            samples.append((str(img_path), source_id, img_path.name))
    del img_paths

    # update .names file in case of new classes
    if (class_names and len(class_mapping) != len(class_names)) or not class_names:
        class_names = [class_name for class_name in class_mapping.keys()]
        if is_numeric:
            class_names = sorted(class_names, key=int)
        write_names_file(class_names, class_file)

    if FLAGS.output_path:
        output_path = Path(FLAGS.output_path).expanduser().resolve()
    else:
        output_path = class_file.parent

    if FLAGS.onnx_model:
        if output_path.is_dir():
            output_path = output_path / f"embedding_{dataset_name}.json"

    if FLAGS.onnx_model:
        generate_embedding(samples, output_path, FLAGS.onnx_model)
        return

    num_samples = len(samples)
    if binary_img:
        num_files = (num_samples + num_records_per_file - 1) // num_records_per_file
    else:
        num_files = 1  # Don't separate files if binary_img is False
        num_records_per_file = num_samples
        output_filename = output_path / f"{dataset_name}.tfrecord"
    logging.info(
        f'Found {num_samples} samples. Writing {num_files} TFRecord file(s) to {output_path}'
    )
    for file_index in range(num_files):
        start_index = file_index * num_records_per_file
        end_index = min((file_index + 1) * num_records_per_file, num_samples)
        if num_files > 1:
            output_filename = output_path / f"{dataset_name}_{file_index}.tfrecord"
        with tf.io.TFRecordWriter(str(output_filename)) as writer:
            for img_str, id_name, filename in tqdm.tqdm(samples[start_index:end_index]):
                write_tfrecord_file(writer, img_str, id_name, filename)


if __name__ == '__main__':
    app.run(main)
