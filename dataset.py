import tensorflow as tf

def _parse_function(filename, label):
    image_contents = tf.read_file(filename)                   # read img to string
    image_decoded = tf.image.decode_image(image_contents)     # decode img to uint8 tensor
    image_resized = tf.image.resize_images(image_decoded, [28, 28])   # resize img
    return image_resized, label

def load_dataset(file_path_list, label_list, batchsize=32, repeat=10):

    assert len(file_path_list) == len(label_list), "file_path_list and label_list have inconsistent shape"

    filenames = tf.constant(file_path_list)
    labels = tf.constant(label_list)

    dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))   # (filename, label)
    dataset = dataset.map(_parse_function)  #(image_resized, label)

    # element in dataset: (image_resized_batch, label_batch)

    dataset = dataset.shuffle(buffersize=1000).batch(batchsize).repeat(repeat)
    # shape of image_resized_batch is(batch_size, img_w, img_h, img channal), shape pf label_batch is(batch_size, )

    return dataset