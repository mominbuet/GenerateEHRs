#! usr/bin/env python3
# -*- coding:utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os
import pickle
import time

import tensorflow as tf
from absl import flags, logging
from tensorflow.contrib.feature_column import sequence_numeric_column

import metrics
from bert import modeling
from bert import optimization
from bert import tokenization

FLAGS = flags.FLAGS

## Required parameters


flags.DEFINE_integer(
    "training_run_count", 1,
    "total number of training count"
)

flags.DEFINE_string(
    "data_dir", None,
    "The input data dir. Should contain the .tsv files (or other data files) "
    "for the task.")

flags.DEFINE_string(
    "bert_config_file", None,
    "The config json file corresponding to the pre-trained BERT model. "
    "This specifies the model architecture.")

flags.DEFINE_string("task_name", None, "The name of the task to train.")

flags.DEFINE_string("vocab_file", None,
                    "The vocabulary file that the BERT model was trained on.")

flags.DEFINE_string(
    "output_dir", None,
    "The output directory where the model checkpoints will be written.")

## Other parameters

flags.DEFINE_string(
    "init_checkpoint", None,
    "Initial checkpoint (usually from a pre-trained BERT model).")

# if you download cased checkpoint you should use "False",if uncased you should use
# "True"
# if we used in bio-medical field，don't do lower case would be better!

flags.DEFINE_bool(
    "do_lower_case", True,
    "Whether to lower case the input text. Should be True for uncased "
    "models and False for cased models.")

flags.DEFINE_integer(
    "max_seq_length", 128,
    "The maximum total input sequence length after WordPiece tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded.")

flags.DEFINE_bool("do_train", True, "Whether to run training.")

flags.DEFINE_bool("do_eval", False, "Whether to run eval on the dev set.")

flags.DEFINE_bool(
    "do_predict", False,
    "Whether to run the model in inference mode on the test set.")

flags.DEFINE_integer("train_batch_size", 32, "Total batch size for training.")

flags.DEFINE_integer("eval_batch_size", 8, "Total batch size for eval.")

flags.DEFINE_integer("predict_batch_size", 8, "Total batch size for predict.")

flags.DEFINE_float("learning_rate", 5e-5, "The initial learning rate for Adam.")

flags.DEFINE_float("num_train_epochs", 3.0,
                   "Total number of training epochs to perform.")

flags.DEFINE_float(
    "warmup_proportion", 0.1,
    "Proportion of training to perform linear learning rate warmup for. "
    "E.g., 0.1 = 10% of training.")

flags.DEFINE_integer("save_checkpoints_steps", 250,
                     "How often to save the model checkpoint.")

flags.DEFINE_integer("iterations_per_loop", 1000,
                     "How many steps to make in each estimator call.")

flags.DEFINE_bool("use_tpu", False, "Whether to use TPU or GPU/CPU.")

flags.DEFINE_string(
    "tpu_name", None,
    "The Cloud TPU to use for training. This should be either the name "
    "used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 "
    "url.")

flags.DEFINE_string(
    "tpu_zone", None,
    "[Optional] GCE zone where the Cloud TPU is located in. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

flags.DEFINE_string(
    "gcp_project", None,
    "[Optional] Project name for the Cloud TPU-enabled project. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

flags.DEFINE_string("master", None, "[Optional] TensorFlow master URL.")

flags.DEFINE_integer(
    "num_tpu_cores", 8,
    "Only used if `use_tpu` is True. Total number of TPU cores to use.")

flags.DEFINE_string("middle_output", "middle_data", "Dir was used to store middle data!")
flags.DEFINE_string("crf", "True", "use crf!")
flags.DEFINE_float("penalty", 0.0, "penalize the loss")


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text, label=None, start=None, document=None):
        """Constructs a InputExample.

        Args:
          guid: Unique id for the example.
          text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
          label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text = text
        self.label = label
        self.start = start
        self.document = document


class PaddingInputExample(object):
    """Fake example so the num input examples is a multiple of the batch size.

    When running eval/predict on the TPU, we need to pad the number of examples
    to be a multiple of the batch size, because the TPU requires a fixed batch
    size. The alternative is to drop the last batch, which is bad because it means
    the entire output data won't be generated.

    We use this class instead of `None` because treating `None` as padding
    battches could cause silent errors.
    """


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self,
                 input_ids,
                 mask,
                 segment_ids,
                 label_ids, binary_label_ids,
                 is_real_example=True):
        self.input_ids = input_ids
        self.mask = mask
        self.segment_ids = segment_ids
        self.label_ids = label_ids
        self.binary_label_ids = binary_label_ids
        self.is_real_example = is_real_example


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_data(cls, input_file):
        """Read a BIO data!"""
        rf = open(input_file, 'r')
        lines = [];
        words = [];
        labels = [];
        starts = [];
        documents = [];
        for line in rf:

            word = line.strip().split(' ')[0]
            label = line.strip().split(' ')[-1]
            start = line.strip().split(' ')[2] if len(line.strip().split(' ')) > 3 else '#'
            document_id = line.strip().split(' ')[-5] if len(line.strip().split(' ')) > 5 else None
            # here we dont do "DOCSTART" check
            i = 0
            if len(line.strip()) == 0 and words[-1] == '.':
                l = ' '.join([label for label in labels if len(label) > 0])
                w = ' '.join([word for word in words if len(word) > 0])
                s = ' '.join([start for start in starts if start is not '#'])
                d = ' '.join([document_id for document_id in documents if document_id is not None])
                lines.append((l, w, s, d))

                words = []
                labels = []
                starts = []
                documents = []

            words.append(word)
            labels.append(label)
            starts.append(start)
            documents.append(document_id)
        rf.close()
        return lines

    @classmethod
    def _read_single_line(cls, input_string):
        lines = [];
        l = ' '.join(['O' for l in input_string.split(' ')])
        w = ' '.join([w for w in input_string.split(' ')])
        lines.append((l, w))
        return lines

    @classmethod
    def _read_words_labels(cls, input_file):
        """Reads a BIO data."""
        with open(input_file) as f:
            lines = []
            words = []
            labels = set()
            for line in f:
                contends = line.strip()
                word = line.strip().split(' ')[0]
                label = line.strip().split(' ')[-1]
                if contends.startswith("-DOCSTART-"):
                    continue
                if (len(label) > 0 and len(word) > 0):
                    words.append(word)
                    labels.add(label)

            final_labels = labels
            return labels


class NerProcessor(DataProcessor):
    def get_train_examples(self, data_dir):
        return self._create_example(
            self._read_data(os.path.join(data_dir, "train.txt")), "train"
        )

    def get_dev_examples(self, data_dir):
        return self._create_example(
            self._read_data(os.path.join(data_dir, "dev.txt")), "dev"
        )

        # return self._create_example(
        #     self._read_data(os.path.join(data_dir, "test.txt")), "dev"
        # )

    def get_test_examples(self, data_dir):
        return self._create_example(
            self._read_data(os.path.join(data_dir, "test.txt")), "test"
        )

    def get_single_example(self, input_string):
        return self._create_example(
            self._read_single_line(input_string), "test")

    def get_labels(self):
        """
        here "X" used to represent "##eer","##soo" and so on!
        "[PAD]" for padding
        :return:
        """
        # return ["[PAD]","B-MISC", "I-MISC", "O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "X","[CLS]","[SEP]"]

        train_labels = self._read_words_labels(os.path.join(FLAGS.data_dir, "train.txt"))
        dev_labels = self._read_words_labels(os.path.join(FLAGS.data_dir, "dev.txt"))
        test_labels = self._read_words_labels(os.path.join(FLAGS.data_dir, "test.txt"))
        merged_labels = list(set().union(train_labels, dev_labels, test_labels))
        merged_labels.append('X')
        merged_labels.append('[CLS]')
        merged_labels.append('[SEP]')
        merged_labels.append('[PAD]')
        return merged_labels

    def _create_example(self, lines, set_type):
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            texts = tokenization.convert_to_unicode(line[1])
            labels = tokenization.convert_to_unicode(line[0])
            examples.append(InputExample(guid=guid, text=texts, label=labels, start=line[2], document=line[3]))
        return examples


def convert_single_example(ex_index, example, label_list, max_seq_length, tokenizer, mode):
    """
    :param ex_index: example num
    :param example:
    :param label_list: all labels
    :param max_seq_length:
    :param tokenizer: WordPiece tokenization
    :param mode:
    :return: feature

    IN this part we should rebuild input sentences to the following format.
    example:[Jim,Hen,##son,was,a,puppet,##eer]
    labels: [I-PER,I-PER,X,O,O,O,X]

    """
    label_map = collections.OrderedDict()
    label_list.sort()
    # here start with zero this means that "[PAD]" is zero
    for (i, label) in enumerate(label_list):
        label_map[label] = i
    with open(FLAGS.middle_output + "/label2id.pkl", 'wb') as w:
        pickle.dump(label_map, w)
    textlist = example.text.split(' ')
    labellist = example.label.split(' ')
    start_position = example.start.split(' ')
    document_list = example.document.split(' ')

    tokens = []
    labels = []
    positions = []
    document_ids = []
    for i, (word, label, position, document) in enumerate(zip(textlist, labellist, start_position, document_list)):
        token = tokenizer.tokenize(word)
        tokens.extend(token)
        for i, _ in enumerate(token):
            if i == 0:
                labels.append(label)
                positions.append(position)
                document_ids.append(document)
            else:
                labels.append("X")
                positions.append('-1')
                document_ids.append('-1')
    # only Account for [CLS] with "- 1".
    if len(tokens) >= max_seq_length - 1:
        tokens = tokens[0:(max_seq_length - 1)]
        labels = labels[0:(max_seq_length - 1)]
    ntokens = []
    segment_ids = []
    label_ids = []
    binary_label_ids = []
    npositions = []
    ndocuments = []
    ntokens.append("[CLS]")
    npositions.append('-1')
    ndocuments.append('-1')
    segment_ids.append(0)
    label_ids.append(label_map["[CLS]"])
    binary_label_ids.append(label_map["[CLS]"])
    for i, token in enumerate(tokens):
        ntokens.append(token)
        npositions.append(positions[i])
        ndocuments.append(document_ids[i])
        segment_ids.append(0)
        label_ids.append(label_map[labels[i]])
        binary_label_ids.append(0 if labels[i] == 'O' else 1)
    # after that we don't add "[SEP]" because we want a sentence don't have
    # stop tag, because i think its not very necessary.
    # or if add "[SEP]" the model even will cause problem, special the crf layer was used.
    input_ids = tokenizer.convert_tokens_to_ids(ntokens)
    mask = [1] * len(input_ids)
    # use zero to padding and you should
    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        mask.append(0)
        segment_ids.append(0)
        label_ids.append(0)
        binary_label_ids.append(0)
        ntokens.append("[PAD]")
        npositions.append('-1')
        ndocuments.append('-1')
    assert len(input_ids) == max_seq_length
    assert len(mask) == max_seq_length
    assert len(segment_ids) == max_seq_length
    assert len(label_ids) == max_seq_length
    assert len(binary_label_ids) == max_seq_length
    # assert len(npositions) == max_seq_length
    assert len(ndocuments) == max_seq_length
    if ex_index < 3:
        logging.info("*** Example ***")
        logging.info("guid: %s" % (example.guid))
        logging.info("tokens: %s" % " ".join(
            [tokenization.printable_text(x) for x in tokens]))
        logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
        logging.info("input_mask: %s" % " ".join([str(x) for x in mask]))
        logging.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
        logging.info("label_ids: %s" % " ".join([str(x) for x in label_ids]))
        logging.info("binary_label_ids: %s" % " ".join([str(x) for x in binary_label_ids]))

    feature = InputFeatures(
        input_ids=input_ids,
        mask=mask,
        segment_ids=segment_ids,
        label_ids=label_ids,
        binary_label_ids=binary_label_ids,
    )
    # we need ntokens because if we do predict it can help us return to original token.
    return feature, ntokens, label_ids, npositions, ndocuments


def filed_based_convert_examples_to_features(examples, label_list, max_seq_length, tokenizer, output_file, mode=None):
    writer = tf.python_io.TFRecordWriter(output_file)
    batch_tokens = []
    batch_labels = []
    batch_start_positions = []
    batch_document_ids = []
    for (ex_index, example) in enumerate(examples):
        # if ex_index % 5000 == 0:
        #     logging.info("Writing example %d of %d" % (ex_index, len(examples)))
        feature, ntokens, label_ids, start_positions, document_ids = convert_single_example(ex_index, example,
                                                                                            label_list,
                                                                                            max_seq_length, tokenizer,
                                                                                            mode)
        batch_tokens.extend(ntokens)
        batch_labels.extend(label_ids)
        batch_start_positions.extend(start_positions)
        batch_document_ids.extend(document_ids)

        def create_int_feature(values):
            f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
            return f

        features = collections.OrderedDict()
        features["input_ids"] = create_int_feature(feature.input_ids)
        features["mask"] = create_int_feature(feature.mask)
        features["segment_ids"] = create_int_feature(feature.segment_ids)
        features["label_ids"] = create_int_feature(feature.label_ids)
        features["binary_label_ids"] = create_int_feature(feature.binary_label_ids)

        tf_example = tf.train.Example(features=tf.train.Features(feature=features))
        writer.write(tf_example.SerializeToString())
    # sentence token in each batch
    writer.close()
    return batch_tokens, batch_labels, batch_start_positions, batch_document_ids


def file_based_input_fn_builder(input_file, seq_length, is_training, drop_remainder):
    name_to_features = {
        "input_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "mask": tf.FixedLenFeature([seq_length], tf.int64),
        "segment_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "label_ids": tf.FixedLenFeature([seq_length], tf.int64),

    }

    def _decode_record(record, name_to_features):
        example = tf.parse_single_example(record, name_to_features)
        for name in list(example.keys()):
            t = example[name]
            if t.dtype == tf.int64:
                t = tf.to_int32(t)
            example[name] = t
        return example

    def input_fn(params):
        batch_size = params["batch_size"]
        d = tf.data.TFRecordDataset(input_file)
        if is_training:
            d = d.repeat()
            d = d.shuffle(buffer_size=100)
        d = d.apply(tf.contrib.data.map_and_batch(  # changed from tf.data.experimental.
            lambda record: _decode_record(record, name_to_features),
            batch_size=batch_size,
            drop_remainder=drop_remainder
        ))
        return d

    return input_fn


# all above are related to data preprocess
# all above are related to data preprocess
# Following i about the model

def hidden2tag(hiddenlayer, numclass):
    linear = tf.keras.layers.Dense(numclass, activation=None)
    return linear(hiddenlayer)


def crf_loss(logits, labels, mask, num_labels, mask2len):
    """

    :param logits:
    :param labels:
    :param mask2len:each sample's length
    :return:
    """
    # TODO
    with tf.variable_scope("crf_loss"):
        trans = tf.get_variable(
            "transition",
            shape=[num_labels, num_labels],
            initializer=tf.contrib.layers.xavier_initializer()
        )

    log_likelihood, transition = tf.contrib.crf.crf_log_likelihood(logits, labels, transition_params=trans,
                                                                   sequence_lengths=mask2len)
    loss = tf.reduce_mean(-log_likelihood)  # tf.math.reduce_mean
    return loss, transition


def softmax_layer(logits, labels, num_labels, mask):
    logits = tf.reshape(logits, [-1, num_labels])
    labels = tf.reshape(labels, [-1])
    mask = tf.cast(mask, dtype=tf.float32)
    one_hot_labels = tf.one_hot(labels, depth=num_labels, dtype=tf.float32)
    loss = tf.losses.softmax_cross_entropy(logits=logits, onehot_labels=one_hot_labels)
    loss *= tf.reshape(mask, [-1])
    loss = tf.reduce_sum(loss)
    total_size = tf.reduce_sum(mask)
    total_size += 1e-12  # to avoid division by 0 for all-0 weights
    loss /= total_size
    # predict not mask we could filtered it in the prediction part.
    probabilities = tf.math.softmax(logits, axis=-1)
    predict = tf.math.argmax(probabilities, axis=-1)
    return loss, predict


def create_model(bert_config, is_training, input_ids, mask,
                 segment_ids, labels, num_labels, use_one_hot_embeddings):
    model = modeling.BertModel(
        config=bert_config,
        is_training=is_training,
        input_ids=input_ids,
        input_mask=mask,
        token_type_ids=segment_ids,
        use_one_hot_embeddings=use_one_hot_embeddings
    )

    output_layer = model.get_sequence_output()
    # output_layer shape is
    if is_training:
        output_layer = tf.keras.layers.Dropout(rate=0.5)(output_layer)
    logits = hidden2tag(output_layer, num_labels)
    # TODO test shape
    logits = tf.reshape(logits, [-1, FLAGS.max_seq_length, num_labels])
    if FLAGS.crf:
        mask2len = tf.reduce_sum(mask, axis=1)
        loss, trans = crf_loss(logits, labels, mask, num_labels, mask2len)
        predict, viterbi_score = tf.contrib.crf.crf_decode(logits, trans, mask2len)
        return (loss, logits, predict)

    else:
        loss, predict = softmax_layer(logits, labels, num_labels, mask)

        return (loss, logits, predict)


def model_fn_builder(bert_config, num_labels, init_checkpoint, learning_rate,
                     num_train_steps, num_warmup_steps, use_tpu,
                     use_one_hot_embeddings):
    def model_fn(features, labels, mode, params):
        logging.info("*** Features ***")
        for name in sorted(features.keys()):
            logging.info("  name = %s, shape = %s" % (name, features[name].shape))
        input_ids = features["input_ids"]
        mask = features["mask"]
        segment_ids = features["segment_ids"]
        label_ids = features["label_ids"]
        is_training = (mode == tf.estimator.ModeKeys.TRAIN)
        if FLAGS.crf:
            (total_loss, logits, predicts) = create_model(bert_config, is_training, input_ids,
                                                          mask, segment_ids, label_ids, num_labels,
                                                          use_one_hot_embeddings)

        else:
            (total_loss, logits, predicts) = create_model(bert_config, is_training, input_ids,
                                                          mask, segment_ids, label_ids, num_labels,
                                                          use_one_hot_embeddings)
        tvars = tf.trainable_variables()
        scaffold_fn = None
        initialized_variable_names = None
        if init_checkpoint:
            (assignment_map, initialized_variable_names) = modeling.get_assignment_map_from_checkpoint(tvars,
                                                                                                       init_checkpoint)
            tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
            if use_tpu:
                def tpu_scaffold():
                    tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
                    return tf.train.Scaffold()

                scaffold_fn = tpu_scaffold
            else:

                tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
        logging.info("**** Trainable Variables ****")
        for var in tvars:
            init_string = ""
            if var.name in initialized_variable_names:
                init_string = ", *INIT_FROM_CKPT*"
            logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                         init_string)

        if mode == tf.estimator.ModeKeys.TRAIN:
            precision, RHO = None, None

            if FLAGS.penalty:
                # with open(FLAGS.middle_output + '/label2id.pkl', 'rb') as rf:
                #     label2id = pickle.load(rf)
                # id2label = {value: key for key, value in label2id.items()}

                # phi_label_ids = tf.not_equal(label_ids, label2id['O'])
                #
                # correct = tf.equal(predicts, label_ids)
                # ture_positive = tf.equal(correct, phi_label_ids)
                # true_positive_count = tf.shape(tf.where(tf.equal(ture_positive, True)))[0]
                # tf.summary.scalar('true_positive', true_positive_count)
                #
                # in_correct = tf.not_equal(predicts, label_ids)
                # false_negative = tf.equal(in_correct, phi_label_ids)
                # false_negative_count = tf.shape(tf.where(tf.equal(false_negative, True)))[0]
                # tf.summary.scalar('false_negative', false_negative_count)
                predictions = tf.argmax(logits, axis=-1, output_type=tf.int32)  # tf.math.argmax
                cmetrics = metrics.streaming_confusion_matrix_with_current_cm(label_ids, predictions, num_labels,
                                                                              weights=mask)
                # total_cm = cmetrics[0]
                current_cm = cmetrics[2]
                # precisions,recall,_ = metrics.calculate(current_cm,num_labels-1)
                # tf.summary.scalar('precisions',precisions)
                # tf.summary.scalar('recall', recall)
                tf.summary.histogram('predictions', predictions)
                tf.summary.histogram('logits', logits)
                tf.summary.histogram('label_ids', label_ids)
                TP = tf.Variable(0, dtype=tf.float32)
                FN = tf.Variable(0, dtype=tf.float32)
                FP = tf.Variable(0, dtype=tf.float32)
                for i in range(num_labels - 1):
                    TP = TP + current_cm[i][i]
                    for j in range(num_labels - 1):
                        if i != j:
                            FN = FN + current_cm[j][i]
                            FP = FP + current_cm[i][j]

                # FN = tf.metrics.false_negatives(label_ids,predicts)[0]

                # TP = tf.metrics.true_positives(label_ids, predicts)[0]
                tf.summary.scalar('true_positive', TP)
                tf.summary.scalar('false_negetive', FN)
                # # FNR = tf.cast(false_negative_count / (false_negative_count + true_positive_count + 1), dtype=tf.float32)
                FNR = FN / (1 + FN + TP)
                precision = TP / (TP + FP)
                # FNR = false_negative_count / (false_negative_count + true_positive_count + 1)
                # # tf.cast(FNR,tf.float32)
                RHO = FNR * (FLAGS.penalty - 1) + 1  # just skipping this one
                # # RHO = tf.cast(RHO,tf.float32)
                #
                tf.summary.scalar('FNR', FNR)
                tf.summary.scalar('penalizing_factor', RHO)
                total_loss = total_loss * RHO  # tf.multiply(total_loss, RHO)

            train_op = optimization.create_optimizer(total_loss, learning_rate, num_train_steps, num_warmup_steps,
                                                     use_tpu, RHO)
            logging_hook = tf.train.LoggingTensorHook({"FNR": FNR, "precision": precision, "loss": total_loss},
                                                      every_n_iter=50)
            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode=mode,
                training_hooks=[logging_hook],
                loss=total_loss,
                train_op=train_op,
                scaffold_fn=scaffold_fn)

        elif mode == tf.estimator.ModeKeys.EVAL:
            def metric_fn(label_ids, logits, num_labels, mask):
                predictions = tf.argmax(logits, axis=-1, output_type=tf.int32)  # tf.math.argmax
                cm = metrics.streaming_confusion_matrix(label_ids, predictions, num_labels - 1, weights=mask)
                # _, r, _ = metrics.calculate(cm, num_labels - 1)
                # return r
                return {
                    "confusion_matrix": cm
                }
                #

            eval_metrics = (metric_fn, [label_ids, logits, num_labels, mask])
            # logging_hook = tf.train.LoggingTensorHook({"eval recall": metric_fn(label_ids, logits, num_labels, mask)},every_n_iter=1)
            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode=mode,
                loss=total_loss,
                # evaluation_hooks=[logging_hook],
                eval_metrics=eval_metrics,
                scaffold_fn=scaffold_fn)
        else:
            predictions = {
                "predicted_logits": predicts
            }
            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode=mode, predictions=predicts, scaffold_fn=scaffold_fn
            )
        return output_spec

    return model_fn


def _write_base(batch_tokens, id2label, prediction, batch_labels, wf, i, batch_start_positions, batch_document_ids):
    token = batch_tokens[i]
    position = batch_start_positions[i]
    document_id = batch_document_ids[i]
    predict = id2label[prediction]
    true_l = id2label[batch_labels[i]]
    if token != "[PAD]" and token != "[CLS]" and true_l != "X":
        #
        if predict == "X" and not predict.startswith("##"):
            predict = "O"
        line = "{}\t{}\t{}\t{}\t{}\n".format(token, document_id, position, predict, true_l)
        wf.write(line)


def Writer(output_predict_file, result, batch_tokens, batch_labels, batch_start_positions, batch_document_ids,
           id2label):
    with open(output_predict_file, 'w') as wf:

        if FLAGS.crf:
            predictions = []
            for m, pred in enumerate(result):
                predictions.extend(pred)
            for i, prediction in enumerate(predictions):
                _write_base(batch_tokens, id2label, prediction, batch_labels, wf, i, batch_start_positions,
                            batch_document_ids)

        else:
            for i, prediction in enumerate(result):
                _write_base(batch_tokens, id2label, prediction, batch_labels, wf, i, batch_start_positions,
                            batch_document_ids)


def main(_):
    if not os.path.exists(FLAGS.output_dir):
        os.makedirs(FLAGS.output_dir)

    logging.set_verbosity(logging.INFO)
    processors = {"ner": NerProcessor}
    if not FLAGS.do_train and not FLAGS.do_eval and not FLAGS.do_predict:
        raise ValueError("At least one of `do_train` or `do_eval` or `do_predict` must be True.")
    bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)
    if FLAGS.max_seq_length > bert_config.max_position_embeddings:
        raise ValueError(
            "Cannot use sequence length %d because the BERT model "
            "was only trained up to sequence length %d" %
            (FLAGS.max_seq_length, bert_config.max_position_embeddings))
    task_name = FLAGS.task_name.lower()
    if task_name not in processors:
        raise ValueError("Task not found: %s" % (task_name))
    processor = processors[task_name]()

    label_list = processor.get_labels()

    tokenizer = tokenization.FullTokenizer(
        vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)
    tpu_cluster_resolver = None
    if FLAGS.use_tpu and FLAGS.tpu_name:
        tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
            FLAGS.tpu_name, zone=FLAGS.tpu_zone, project=FLAGS.gcp_project)
    is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2
    config = tf.ConfigProto()
    if FLAGS.do_train:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        config.log_device_placement = False
        config.gpu_options.per_process_gpu_memory_fraction = 0.9
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        config.gpu_options.allow_growth = False
        config.gpu_options.per_process_gpu_memory_fraction = 0.1
    run_config = tf.contrib.tpu.RunConfig(
        cluster=tpu_cluster_resolver,
        master=FLAGS.master,
        model_dir=FLAGS.output_dir,
        save_checkpoints_steps=FLAGS.save_checkpoints_steps,
        session_config=config,
        keep_checkpoint_max=3,
        tpu_config=tf.contrib.tpu.TPUConfig(
            iterations_per_loop=FLAGS.iterations_per_loop,
            num_shards=FLAGS.num_tpu_cores,
            per_host_input_for_training=is_per_host))
    train_examples = None
    num_train_steps = None
    num_warmup_steps = None

    if FLAGS.do_train:
        train_examples = processor.get_train_examples(FLAGS.data_dir)

        num_train_steps = int(
            len(train_examples) / FLAGS.train_batch_size * FLAGS.num_train_epochs * FLAGS.training_run_count)
        num_warmup_steps = int(num_train_steps * FLAGS.warmup_proportion)
    model_fn = model_fn_builder(
        bert_config=bert_config,
        num_labels=len(label_list),
        init_checkpoint=FLAGS.init_checkpoint,
        learning_rate=FLAGS.learning_rate,
        num_train_steps=num_train_steps,
        num_warmup_steps=num_warmup_steps,
        use_tpu=FLAGS.use_tpu,
        use_one_hot_embeddings=FLAGS.use_tpu)
    estimator = tf.contrib.tpu.TPUEstimator(
        use_tpu=FLAGS.use_tpu,
        model_fn=model_fn,
        config=run_config,
        train_batch_size=FLAGS.train_batch_size,
        predict_batch_size=FLAGS.predict_batch_size,
        eval_batch_size=FLAGS.eval_batch_size
    )
    # if FLAGS.do_train and FLAGS.do_eval:
    #     train_file = os.path.join(FLAGS.output_dir, "train.tf_record")
    #     _, _, _, _ = filed_based_convert_examples_to_features(
    #         train_examples, label_list, FLAGS.max_seq_length, tokenizer, train_file)
    #     logging.info("***** Running training *****")
    #     logging.info("  Num examples = %d", len(train_examples))
    #     logging.info("  Batch size = %d", FLAGS.train_batch_size)
    #     logging.info("  Num steps = %d", num_train_steps)
    #     train_input_fn = file_based_input_fn_builder(
    #         input_file=train_file,
    #         seq_length=FLAGS.max_seq_length,
    #         is_training=True,
    #         drop_remainder=True)
    #
    #     # eval_examples = processor.get_dev_examples(FLAGS.data_dir)
    #     # eval_file = os.path.join(FLAGS.output_dir, "eval.tf_record")
    #     # batch_tokens, batch_labels, batch_start_positions, batch_document_ids = filed_based_convert_examples_to_features(
    #     #     eval_examples, label_list, FLAGS.max_seq_length, tokenizer, eval_file)
    #     # eval_input_fn = file_based_input_fn_builder(
    #     #     input_file=eval_file,
    #     #     seq_length=FLAGS.max_seq_length,
    #     #     is_training=False,
    #     #     drop_remainder=False)
    #
    #     # train_spec = tf.estimator.TrainSpec(input_fn=train_input_fn, max_steps=num_train_steps)
    #     # eval_spec = tf.estimator.EvalSpec(input_fn=eval_input_fn, steps=10)
    #     estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)
    #     # tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)

    if FLAGS.do_train:
        train_file = os.path.join(FLAGS.output_dir, "train.tf_record")
        _, _, _, _ = filed_based_convert_examples_to_features(
            train_examples, label_list, FLAGS.max_seq_length, tokenizer, train_file)
        logging.info("***** Running training *****")
        logging.info("  Num examples = %d", len(train_examples))
        logging.info("  Batch size = %d", FLAGS.train_batch_size)
        logging.info("  Num steps = %d", num_train_steps)
        train_input_fn = file_based_input_fn_builder(
            input_file=train_file,
            seq_length=FLAGS.max_seq_length,
            is_training=True,
            drop_remainder=True)
        start_time = time.time()
        estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)
        estimated_time = time.time() - start_time
        print('Training time {} Seconds'.format(estimated_time))

    if FLAGS.do_eval:
        # # run evaluation on training set
        #
        # logging.info("***** Running evaluation on training *****")
        # logging.info("  Num examples = %d", len(train_examples))
        # logging.info("  Batch size = %d", FLAGS.train_batch_size)
        # # if FLAGS.use_tpu:
        # #     eval_steps = int(len(eval_examples) / FLAGS.)
        # # eval_drop_remainder = True if FLAGS.use_tpu else False
        #
        # training_result = estimator.evaluate(input_fn=train_input_fn)
        # output_training_file = os.path.join(FLAGS.output_dir, "training_result.txt")
        # with open(output_training_file, "w") as wf:
        #     logging.info("***** Training Eval results *****")
        #     confusion_matrix = training_result["confusion_matrix"]
        #     p, r, f = metrics.calculate(confusion_matrix, len(label_list) - 1)
        #     logging.info("***********************************************")
        #     logging.info("********************P = %s*********************", str(p))
        #     logging.info("********************R = %s*********************", str(r))
        #     logging.info("********************F = %s*********************", str(f))
        #     logging.info("***********************************************")
        #     wf.write("***********************************************")
        #     wf.write("********************P =" + str(p) + "*********************\n")
        #     wf.write("********************R =" + str(r) + "*********************\n")
        #     wf.write("********************F =" + str(f) + "*********************\n")

        # run evaluation on dev set

        eval_examples = processor.get_dev_examples(FLAGS.data_dir)
        eval_file = os.path.join(FLAGS.output_dir, "eval.tf_record")
        batch_tokens, batch_labels, batch_start_positions, batch_document_ids = filed_based_convert_examples_to_features(
            eval_examples, label_list, FLAGS.max_seq_length, tokenizer, eval_file)

        logging.info("***** Running evaluation on development *****")
        logging.info("  Num examples = %d", len(eval_examples))
        logging.info("  Batch size = %d", FLAGS.eval_batch_size)
        # if FLAGS.use_tpu:
        #     eval_steps = int(len(eval_examples) / FLAGS.eval_batch_size)
        # eval_drop_remainder = True if FLAGS.use_tpu else False
        eval_input_fn = file_based_input_fn_builder(
            input_file=eval_file,
            seq_length=FLAGS.max_seq_length,
            is_training=False,
            drop_remainder=False)
        result = estimator.evaluate(input_fn=eval_input_fn)
        output_eval_file = os.path.join(FLAGS.output_dir, "eval_results.txt")
        with open(output_eval_file, "w") as wf:
            logging.info("***** Eval results *****")
            confusion_matrix = result["confusion_matrix"]
            p, r, f = metrics.calculate(confusion_matrix, len(label_list) - 1)
            logging.info("***********************************************")
            logging.info("********************P = %s*********************", str(p))
            logging.info("********************R = %s*********************", str(r))
            logging.info("********************F = %s*********************", str(f))
            logging.info("***********************************************")
            wf.write("***********************************************")
            wf.write("********************P =" + str(p) + "*********************")
            wf.write("********************R =" + str(r) + "*********************")
            wf.write("********************F =" + str(f) + "*********************")

    if FLAGS.do_predict:
        with open(FLAGS.middle_output + '/label2id.pkl', 'rb') as rf:
            label2id = pickle.load(rf)
            id2label = {value: key for key, value in label2id.items()}

        predict_examples = processor.get_test_examples(FLAGS.data_dir)

        predict_file = os.path.join(FLAGS.output_dir, "predict.tf_record")
        batch_tokens, batch_labels, batch_start_positions, batch_document_ids = filed_based_convert_examples_to_features(
            predict_examples,
            label_list,
            FLAGS.max_seq_length,
            tokenizer,
            predict_file)

        logging.info("***** Running prediction*****")
        logging.info("  Num examples = %d", len(predict_examples))
        logging.info("  Batch size = %d", FLAGS.predict_batch_size)

        predict_input_fn = file_based_input_fn_builder(
            input_file=predict_file,
            seq_length=FLAGS.max_seq_length,
            is_training=False,
            drop_remainder=False)
        predict_start_time = time.time()

        result = estimator.predict(input_fn=predict_input_fn)

        predict_estimated_time = time.time() - predict_start_time
        print('Prediction time {} Seconds'.format(predict_estimated_time))

        output_predict_file = os.path.join(FLAGS.output_dir, "label_test.txt")
        # here if the tag is "X" means it belong to its before token, here for convenient evaluate use
        # conlleval.pl we  discarding it directly
        Writer(output_predict_file, result, batch_tokens, batch_labels, batch_start_positions, batch_document_ids,
               id2label)

        # PREDICT FOR TRAIN DATA

        train_examples = processor.get_train_examples(FLAGS.data_dir)

        train_predict_file = os.path.join(FLAGS.output_dir, "train.tf_record")
        batch_tokens, batch_labels, batch_start_positions, batch_document_ids = filed_based_convert_examples_to_features(
            train_examples,
            label_list,
            FLAGS.max_seq_length,
            tokenizer,
            train_predict_file)

        logging.info("***** Running prediction for training data*****")
        logging.info("  Num examples = %d", len(predict_examples))
        logging.info("  Batch size = %d", FLAGS.predict_batch_size)

        train_predict_input_fn = file_based_input_fn_builder(
            input_file=train_predict_file,
            seq_length=FLAGS.max_seq_length,
            is_training=False,
            drop_remainder=False)

        result = estimator.predict(input_fn=train_predict_input_fn)
        output_predict_file = os.path.join(FLAGS.output_dir, "label_train.txt")
        # here if the tag is "X" means it belong to its before token, here for convenient evaluate use
        # conlleval.pl we  discarding it directly
        Writer(output_predict_file, result, batch_tokens, batch_labels, batch_start_positions, batch_document_ids,
               id2label)

    # SAVE MODEL FOR SERVING

    feature_columns_keys = ['input_ids', 'mask', 'segment_ids', 'label_ids']
    feature_columns = [sequence_numeric_column(key=key) for key in feature_columns_keys]

    # feature_columns["input_ids"] = sequence_numeric_column('input_ids')
    # feature_columns["mask"] = sequence_numeric_column('mask')
    # feature_columns["segment_ids"] = sequence_numeric_column('segment_ids')
    # feature_columns["label_ids"] = sequence_numeric_column('label_ids')
    def serving_input_fn():
        label_ids = tf.placeholder(tf.int32, [None, FLAGS.max_seq_length], name='label_ids')
        input_ids = tf.placeholder(tf.int32, [None, FLAGS.max_seq_length], name='input_ids')
        input_mask = tf.placeholder(tf.int32, [None, FLAGS.max_seq_length], name='mask')
        segment_ids = tf.placeholder(tf.int32, [None, FLAGS.max_seq_length], name='segment_ids')
        input_fn = tf.estimator.export.build_raw_serving_input_receiver_fn({
            'label_ids': label_ids,
            'input_ids': input_ids,
            'mask': input_mask,
            'segment_ids': segment_ids,
        })()
        return input_fn

    def serving_input_fn_prev():
        with tf.variable_scope("foo"):
            feature_spec = {
                "input_ids": tf.FixedLenFeature([FLAGS.max_seq_length], tf.int64),
                "mask": tf.FixedLenFeature([FLAGS.max_seq_length], tf.int64),
                "segment_ids": tf.FixedLenFeature([FLAGS.max_seq_length], tf.int64),
                "label_ids": tf.FixedLenFeature([FLAGS.max_seq_length], tf.int64),
            }
            serialized_tf_example = tf.placeholder(dtype=tf.string,
                                                   shape=[None],
                                                   name='input_example_tensor')
            receiver_tensors = {'examples': serialized_tf_example}
            features = tf.parse_example(serialized_tf_example, feature_spec)
            return tf.estimator.export.ServingInputReceiver(features, receiver_tensors)

    estimator._export_to_tpu = False  # this is important
    export_dir = estimator.export_savedmodel(FLAGS.output_dir, serving_input_fn)
    print('Exported to {}'.format(export_dir))


if __name__ == "__main__":
    flags.mark_flag_as_required("data_dir")
    flags.mark_flag_as_required("task_name")
    flags.mark_flag_as_required("vocab_file")
    flags.mark_flag_as_required("bert_config_file")
    flags.mark_flag_as_required("output_dir")

    tf.app.run()
