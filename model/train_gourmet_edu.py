import tensorflow as tf
import numpy as np
import h5py
from time import time
from datetime import datetime
from random import shuffle
from functools import reduce
import os

max_seg = 20
max_word = 40

level_class_cnt = 3
test_percentage = 0.2

dropout_rate = tf.Variable(0.5)
hidden_feature_dim = 100
gru_feature_dim = 50
kernel_heights = [3, 4, 5]

batch_size = 256
epochs = 8

main_path = '/home/tim/Documents/NLP/gourmet/test'
w2v_weights_path = main_path + '/weights.npy'
tensorboard_log_dir_train = "/tmp/pycharm_nlp/logs/remake2/g_sen_large_max_" + datetime.now().strftime("%Y%m%d-%H%M%S") + "train"
tensorboard_log_dir_test = "/tmp/pycharm_nlp/logs/remake2/g_sen_large_max_" + datetime.now().strftime("%Y%m%d-%H%M%S") + "test"
input_path = main_path + '/gourmet.hdf5'
model_out_path = main_path + '/model.h5'

sample_amount = 0
mini_batch_cnt = 0
with h5py.File(input_path) as in_file:
    for index in range(len(in_file['label/'].keys())):
        mini_batch_cnt += 1
        sample_amount += len(in_file['label/' + str(index)])
batch_indices = [*range(mini_batch_cnt)]
shuffle(batch_indices)
train_batches = batch_indices[0:int(mini_batch_cnt * (1 - test_percentage))]
test_batches = batch_indices[int(mini_batch_cnt * (1 - test_percentage)):]
w2v = np.load(w2v_weights_path, allow_pickle=True)
w2v_len = w2v.shape[1]


def map_to_w2v(x):
    """ Get the w2v embedding for a specific w2v-id """
    return w2v[int(x)]


def __label_map(raw_label):
    """ Map the 5 label classes to the 3 classes """
    if raw_label < 3:
        return 0
    elif raw_label == 3:
        return 1
    else:
        return 2


def __balance_data(feature_array, label_array):
    """ Balance batches """
    to_balance_indices = np.where(label_array == 2)[0]
    feature_array = np.delete(feature_array, to_balance_indices, axis=0)
    label_array = np.delete(label_array, to_balance_indices, axis=0)

    to_balance_indices = np.where(label_array == 4)[0]
    feature_array = np.delete(feature_array, to_balance_indices, axis=0)
    label_array = np.delete(label_array, to_balance_indices, axis=0)
    return feature_array, label_array


def data_generator(batch_indices, max_seg=max_seg, max_word=max_word, epochs=epochs, use_balance=False):
    """ Generator for the dataset
    
    Generate labels and segments from the batches batch_indices and crop them to the max_seg and max_word sizes
    """
    global batch_size, input_path
    with h5py.File(input_path) as in_file:
        feature_array, label_array = np.zeros((batch_size, max_seg, max_word)), np.zeros((batch_size, 1))
        batch_index = 0
        for _ in range(epochs):
            shuffle(batch_indices)
            for index in batch_indices:
                doc, label = in_file['document/' + str(index)], in_file['label/' + str(index)]
                random_doc_order = [*range(len(doc))]
                shuffle(random_doc_order)
                for i in random_doc_order:
                    feature_array[batch_index] = doc[i][:max_seg, :max_word]
                    label_array[batch_index] = label[i]
                    batch_index += 1
                    if batch_index == batch_size:
                        label_array += 1
                        label_array = label_array.astype(np.int32)
                        if use_balance:
                            feature_array, label_array = __balance_data(feature_array, label_array)
                            label_array = [np.array([__label_map(l[0])]) for l in label_array]
                            yield feature_array, label_array
                        else:
                            label_array = [np.array([__label_map(l[0])]) for l in label_array]
                            yield feature_array, label_array
                        batch_index = 0
                        feature_array, label_array = np.zeros((batch_size, max_seg, max_word)), np.zeros(
                            (batch_size, 1))




def __get_filter_layer(total_dim, target_dim, index):
    """ Slice a piece from one dimension.

    The layer would slice the `index`th dimension from `target_dim` dimension of
    the input tensor, which have `total_dim` dimensions, then squeeze the tensor
    over the sliced dimension.

    Args:
        total_dim (int): The total number of dimensions of the input tensor.
        target_dim (int): The index of the dimension that need to slice.
        index (int): The index of the dimension to keep in the slicing operation.

    Returns:
        (Model): A keras model that implement the operation.
    """
    def tensor_filter(tensor_in):
        nonlocal index
        begin = [0 if i != target_dim else index for i in range(total_dim)]
        size = [-1 if i != target_dim else 1 for i in range(total_dim)]
        return tf.squeeze(tf.slice(tensor_in, begin, size), axis=target_dim)

    return tf.keras.models.Sequential([
        tf.keras.layers.Lambda(tensor_filter)
    ])




def __get_branch_model(input_shape, branch_index, output_shape, submodel, args={}):
    """ Implement `submodel` for each slice of tensor.

    The model would slice its input tensor into pieces using `__get_filter_layer` 
    along `branch_index`th dimension, then for each slice, implement submodel, 
    finally the outputs of different submodels would be concated and reshaped to 
    meet the demand of output.

    Args:
        input_shape tuple(int): The shape of the input tensor.
        branch_index (int): The index of the dimension to slice, start from 0 as 
            sample amount dimension.
        output_shape tuple(int): The shape of the output tensor.
        submodel (Model): The model to apply to different slices.
        args (dict): The argument dictionary for `submodel`.
    """
    model_input = tf.keras.Input(input_shape)
    sliced_inputs = [__get_filter_layer(len(input_shape) + 1, branch_index, i)(model_input)
                     for i in range(input_shape[branch_index - 1])]
    sub_instance = submodel(**args)
    branch_models = [sub_instance(sliced_inputs[i])
                     for i in range(input_shape[branch_index - 1])]
    concated_layers = tf.keras.layers.Concatenate()(branch_models)
    model_output = tf.keras.layers.Reshape(output_shape)(concated_layers)
    if 'name' in args:
        print(model_output)
    return tf.keras.Model(model_input, model_output)





def __get_sentence_encode_unit(input_shape, hidden_feature_dim, kernel_height):
    """ A CNN unit to encode segment with single kernel height.

    The unit would apply a convolution to its input to get a 2-dimensional 
    tensor, then apply max overtime pooling to get a single dimensional tensor.

    Args:
        input_shape ((int, int)): The shape of segment matrix. (word_max, w2v_len)
        hidden_feature_dim (int): The dimension of the hidden feature.
        kernel_height (int): The height of the convolution kernel.

    Returns:
        (Model): The CNN model to encode the segment matrix.
    """
    cnned_height = input_shape[0] - kernel_height + 1
    return tf.keras.models.Sequential([
        tf.keras.layers.Reshape((*input_shape, 1)),
        tf.keras.layers.Conv2D(hidden_feature_dim, (kernel_height, input_shape[1]), kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        # , weights=tf.keras.initializers.uniform(minval=-0.01, maxval=0.01)
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ReLU(),
        tf.keras.layers.Reshape((cnned_height, hidden_feature_dim, 1)),
        tf.keras.layers.MaxPool2D((cnned_height, 1))
    ], name="sentence_encode%s" % kernel_height)




def __get_multi_kernel_encode_unit(input_shape, hidden_feature_dim, kernel_heights):
    """ A CNN unit to encode segment with multiple kernel heights

    The unit would apply operation defined in `__get_sentence_encode_unit` for 
    different kernel heights, then concat the result as a 1-dimensional tensor.

    Args:
        input_shape ((int, int)): The shape of the document. (word_max, w2v_len)
        hidden_feature_dim (int): The dimension of the hidden feature.
        kernel_heights ([int]): The list of the kernel heights.

    Returns:
        (Model): The CNN model to encode the segment matrix.
    """
    model_input = tf.keras.Input(input_shape)
    cnn_layers = [__get_sentence_encode_unit((input_shape), hidden_feature_dim, h)
                  (model_input) for h in kernel_heights]
    concated_layers = tf.keras.layers.Concatenate()(cnn_layers)
    model_output = tf.keras.layers.Flatten()(concated_layers)
    return tf.keras.Model(model_input, model_output)




def __get_seg_classifier_unit(class_cnt, dropout_rate):
    """ The softmax linear classifier for predicting segment sentiment.

    Args:
        class_cnt (int): Number of classes in the classification.
        dropout_rate (int): The drop out rate of the drop out layer.

    Returns:
        (Model): The softmax linear classifier to predict segment sentiment.
    """
    return tf.keras.models.Sequential([
        tf.keras.layers.Dropout(dropout_rate),
        tf.keras.layers.Dense(class_cnt, activation='softmax', kernel_regularizer=tf.keras.regularizers.l2(0.001))
    ])




def __get_attention_unit(gru_feature_dim, dropout_rate):
    """ The unit to get the attention weight for a segment from hidden feature.

    Args:
        gru_feature_dim: The number of out dimensions of GRU layer.
        dropout_rate: The drop out rate of the drop out layer.

    Returns:
        (Model): The model for predicting attention weight for a segment.

    """
    return tf.keras.models.Sequential([
        tf.keras.layers.Dropout(dropout_rate),
        tf.keras.layers.Dense(len(kernel_heights) * hidden_feature_dim, activation='tanh', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        tf.keras.layers.Dense(1, kernel_initializer='glorot_uniform', use_bias=False, kernel_regularizer=tf.keras.regularizers.l2(0.001))
    ], name="attention_unit")





def __get_bidirectional_gru_unit(gru_feature_dim):
    """ A bidirectional-GRU unit to extract the hidden vectors.

    The hidden vectors are used to predict the attention weights of the model.

    Args:
        gru_feature_dim (int): The output dimension of the GRU layer.

    Returns:
        (Model): The bidirectional-GRU unit to predict the hidden vectors.
    """
    return tf.keras.models.Sequential([
        tf.keras.layers.Bidirectional(
            tf.keras.layers.GRU(gru_feature_dim, return_sequences=True, kernel_regularizer=tf.keras.regularizers.l2(0.001)), merge_mode='concat'
            # TODO: are those to separate GRU modules, or is the same reused? (paper says they should be separate)
        )
    ], name="bidirectional")


print('Constructing Model ...', end='')

model_input = tf.keras.Input((max_seg, max_word), dtype=tf.float32)

embedding_layer = tf.keras.layers.Embedding(
    input_dim=w2v.shape[0],
    output_dim=w2v_len,
    weights=[w2v],
    input_length=max_word,
    trainable=False
)(model_input)

encoding_model = __get_branch_model(
    input_shape=(max_seg, max_word, w2v_len),
    branch_index=1,
    output_shape=(max_seg, len(kernel_heights) * hidden_feature_dim),
    submodel=__get_multi_kernel_encode_unit,
    args={
        'input_shape': (max_word, w2v_len),
        'hidden_feature_dim': hidden_feature_dim,
        'kernel_heights': kernel_heights,
    }
)(embedding_layer)

biglu_model = __get_bidirectional_gru_unit(
    gru_feature_dim
)(encoding_model)

attention_model = __get_branch_model(
    input_shape=(max_seg, len(kernel_heights) * hidden_feature_dim),
    branch_index=1,
    output_shape=(max_seg, 1),
    submodel=__get_attention_unit,
    args={
        'gru_feature_dim': gru_feature_dim,
        'dropout_rate': dropout_rate
    }
)(encoding_model)

softmax_model = tf.keras.layers.Softmax(axis=1)(attention_model)

classification_model = __get_branch_model(
    input_shape=(max_seg, len(kernel_heights) * hidden_feature_dim),
    branch_index=1,
    output_shape=(max_seg, level_class_cnt),
    submodel=__get_seg_classifier_unit,
    args={
        'class_cnt': level_class_cnt,
        'dropout_rate': dropout_rate
    }
)(encoding_model)

weighted_layer = tf.keras.layers.Multiply(
)([softmax_model, classification_model])

reduce_layer = tf.keras.layers.Lambda(
    tf.reduce_mean,
    arguments={
        'axis': 1
    }
)(weighted_layer)

model = tf.keras.Model(model_input, reduce_layer)

print('\rModel Constructed. Compiling ...', end='')


def train():
    """ Custom training method. Particular useful for debugging the correctness of the model
    Performs evaluation on the validation set and the training set and outputs tensorboard summaries
    """
    global model
    tf.Graph().as_default()
    tf.device('/gpu:0')
    global_step = tf.Variable(0)
    optimizer = tf.train.AdamOptimizer()
    session = tf.Session()

    ##############################
    # Train -----------------
    ##############################
    train_dropout = tf.assign(dropout_rate, 0.5)
    train_writer = tf.summary.FileWriter(tensorboard_log_dir_train, session.graph)

    dataset = tf.data.Dataset.from_generator(lambda: data_generator(train_batches, use_balance=True),
                                             output_types=(tf.float32, tf.int64))
    dataset.prefetch(3)

    train_iterator = tf.data.Iterator.from_structure(dataset.output_types, dataset.output_shapes)
    train_data_init = train_iterator.make_initializer(dataset)
    session.run(train_data_init)
    features, labels = train_iterator.get_next()

    logits = model(features)
    pred = tf.argmax(logits, 1)

    train_loss = tf.keras.losses.SparseCategoricalCrossentropy()(y_true=labels, y_pred=logits)

    train = optimizer.minimize(train_loss, global_step=global_step)
    labels_temp = tf.squeeze(labels, axis=1)
    labels_temp = tf.cast(labels_temp, tf.int64)
    train_accuracy = tf.reduce_sum(tf.cast(tf.equal(pred, labels_temp), tf.int32)) / tf.size(pred)

    ##############################
    # Validation -----------------
    ##############################
    test_dropout = tf.assign(dropout_rate, 1.0)
    test_writer = tf.summary.FileWriter(tensorboard_log_dir_test, session.graph)
    test_dataset = tf.data.Dataset.from_generator(lambda: data_generator(test_batches, use_balance=True),
                                                  output_types=(tf.float32, tf.int64))
    test_dataset.prefetch(3)

    test_iterator = tf.data.Iterator.from_structure(test_dataset.output_types, test_dataset.output_shapes)
    test_data_init = test_iterator.make_initializer(test_dataset)
    session.run(test_data_init)
    test_features, test_labels = test_iterator.get_next()

    test_logits = model(test_features)
    test_pred = tf.argmax(test_logits, 1)

    test_labels_temp = tf.squeeze(test_labels, axis=1)
    test_labels_temp = tf.cast(test_labels_temp, tf.int64)
    test_accuracy = tf.reduce_sum(tf.cast(tf.equal(test_pred, test_labels_temp), tf.int32)) / tf.size(test_pred)

    session.run(tf.compat.v1.global_variables_initializer())
    inc_step = tf.compat.v1.assign_add(global_step, 1, name='increment')
    steps_per_epoch = int((sample_amount * (1 - test_percentage) // batch_size) - 1)

    # Output of classification layer
    new_model = tf.keras.Model(model.input, classification_model)
    classification_out = new_model(features)

    new_model2 = tf.keras.Model(model.input, biglu_model)
    biglu_out = new_model2(features)

    new_model3 = tf.keras.Model(model.input, softmax_model)
    attention_out = new_model3(features)

    hm = tf.summary.scalar('asdf', tf.constant(2))

    merged = tf.summary.merge([hm])

    accuracies = []
    losses = []
    for i in range(int(epochs * steps_per_epoch)):
        _, loss, logits_res, y, accuracy, prediction, class_out, att_out, bigl_out, feat, sum_merged = session.run(
            [train, train_loss, logits, labels, train_accuracy, pred, classification_out, attention_out, biglu_out,
             features, merged])

        accuracies.append(accuracy)
        losses.append(loss)

        if i % 20 == 0:
            print("Accuracy: %s" % accuracy)
            print("loss: %s " % loss)
            print("Batch: %s " % i)
            a = 0

        if i % 100 == 0:
            print("summary")
            accuracy = np.mean(accuracies)
            loss = np.mean(losses)
            summary = tf.compat.v1.Summary()
            summary.value.add(tag="%sAccuracy", simple_value=accuracy)
            summary.value.add(tag="%sLoss", simple_value=loss)
            the_incremented_step = session.run(inc_step)
            train_writer.add_summary(summary, the_incremented_step)
            train_writer.add_summary(sum_merged, the_incremented_step)
            train_writer.flush()
            accuracies = []
            losses = []

        # Run validation
        if i % 220 == 0 and i != 0:
            class_cnt = 3
            eps = np.finfo(float).eps
            accuracy, precisions, recalls, f1s, losses = [], [], [], [], []
            for j in range(len(test_batches)):
                session.run([test_data_init, test_dropout])
                y, asdf, pred_test, logits_test = session.run([test_labels, test_accuracy, test_pred, test_logits])
                precisions.append([])
                recalls.append([])
                f1s.append([])
                contingency_table = np.zeros((class_cnt, class_cnt))
                for index in range(len(y)):
                    contingency_table[int(y[index][0])][np.argmax(logits_test[index])] += 1
                accuracy.append(np.trace(contingency_table) / len(y))
                for index in range(class_cnt):
                    precisions[j].append(
                        contingency_table[index][index] / (np.sum(contingency_table[:, index]) + eps))
                    recalls[j].append(
                        contingency_table[index][index] / (np.sum(contingency_table[index, :]) + eps))
                    f1s[j].append(
                        2 * precisions[j][-1] * recalls[j][-1] / ((precisions[j][-1] + recalls[j][-1]) + eps))
            precisions = [float(sum(l)) / len(l) for l in zip(*precisions)]
            recalls = [float(sum(l)) / len(l) for l in zip(*recalls)]
            f1s = [float(sum(l)) / len(l) for l in zip(*f1s)]
            print('Accuracy:', round(reduce(lambda x, y: x + y, accuracy) / len(accuracy), 3))
            for index in range(class_cnt):
                print('_____ Class', index, '_____')
                print('Precision\t', round(precisions[index], 3))
                print('Recall\t\t', round(recalls[index], 3))
                print('F1 Score\t', round(f1s[index], 3))

            summary = tf.compat.v1.Summary()
            print(accuracy)
            summary.value.add(tag="%sAccuracy",
                              simple_value=round(reduce(lambda x, y: x + y, accuracy) / len(accuracy), 3))
            summary.value.add(tag="%sF1-Score", simple_value=np.mean(f1s))
            step = session.run(global_step)
            test_writer.add_summary(summary, step)
            test_writer.flush()
            session.run(train_dropout)

train()

model.save(model_out_path)
