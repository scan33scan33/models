##Install and import dependencies
!pip install -q tf-models-official==2.3.0



import os

import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf

import tensorflow_hub as hub

os.environ["TFHUB_CACHE_DIR"] = "gs://nts2020-tpu"

from official import nlp
from official.modeling import tf_utils
from official.nlp import bert

# Load the required submodules
import official.nlp.optimization
import official.nlp.bert.bert_models
import official.nlp.bert.configs
import official.nlp.bert.run_classifier
import official.nlp.bert.tokenization
import official.nlp.data.classifier_data_lib
import official.nlp.modeling.losses
import official.nlp.modeling.models
import official.nlp.modeling.networks

import json


# Initialize TPU first. Initializing TPU after loading the dataset may cause proto error. Therefore, we initialize TPUs first.

try:
  tpu = tf.distribute.cluster_resolver.TPUClusterResolver(tpu='tpu-quickstart')
  tf.config.experimental_connect_to_cluster(tpu)
  tf.tpu.experimental.initialize_tpu_system(tpu)# TPU detection
  print('Running on TPU ', tpu.cluster_spec().as_dict()['worker'])
except ValueError:
  raise BaseException('ERROR: Not connected to a TPU runtime; please see the previous cell in this notebook for instructions!')


# Utility functions to load the classification datasets for XNLI

def single_file_dataset(input_file, name_to_features, num_samples=None):
  """Creates a single-file dataset to be passed for BERT custom training."""
  # For training, we want a lot of parallel reading and shuffling.
  # For eval, we want no shuffling and parallel reading doesn't matter.
  d = tf.data.TFRecordDataset(input_file)
  if num_samples:
    d = d.take(num_samples)
  d = d.map(
      lambda record: decode_record(record, name_to_features),
      num_parallel_calls=tf.data.experimental.AUTOTUNE)

  # When `input_file` is a path to a single file or a list
  # containing a single path, disable auto sharding so that
  # same input file is sent to all workers.
  if isinstance(input_file, str) or len(input_file) == 1:
    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = (
        tf.data.experimental.AutoShardPolicy.OFF)
    d = d.with_options(options)
  return d




def decode_record(record, name_to_features):
  """Decodes a record to a TensorFlow example."""
  example = tf.io.parse_single_example(record, name_to_features)

  # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
  # So cast all int64 to int32.
  for name in list(example.keys()):
    t = example[name]
    if t.dtype == tf.int64:
      t = tf.cast(t, tf.int32)
    example[name] = t

  return example


#create_classifier_dataset has seperate fields called task_id and lang_id. task_id distinguishes between different tasks like XNLI, and PAWS while lang_id distinguishes between different languages in the XNLI task. Tasks with single language (EN assumed to be default) has a lang_id = 0


def create_classifier_dataset(file_path,
                              seq_length,
                              batch_size,
                              task_id,
                              is_training=True,
                              input_pipeline_context=None,
                              label_type=tf.int64,
                              lang_id = 0,
                              include_sample_weights=False,
                              num_samples=None):
  """Creates input dataset from (tf)records files for train/eval."""
  name_to_features = {
      'input_ids': tf.io.FixedLenFeature([seq_length], tf.int64),
      'input_mask': tf.io.FixedLenFeature([seq_length], tf.int64),
      'segment_ids': tf.io.FixedLenFeature([seq_length], tf.int64),
      'label_ids': tf.io.FixedLenFeature([], label_type),
  }
  if include_sample_weights:
    name_to_features['weight'] = tf.io.FixedLenFeature([], tf.float32)
  dataset = single_file_dataset(file_path, name_to_features,
                                num_samples=num_samples)

  # The dataset is always sharded by number of hosts.
  # num_input_pipelines is the number of hosts rather than number of cores.
  if input_pipeline_context and input_pipeline_context.num_input_pipelines > 1:
    dataset = dataset.shard(input_pipeline_context.num_input_pipelines,
                            input_pipeline_context.input_pipeline_id)

  def _select_data_from_record(record):
    x = {
        'input_word_ids': record['input_ids'],
        'input_mask': record['input_mask'],
        'input_type_ids': record['segment_ids'],
         'lang_id' : lang_id
    }
    #pdb.set_trace()
    y = record['label_ids']
    if include_sample_weights:
      w = record['weight']
      return (x, y, w)
    default = tf.constant(-1, dtype=tf.int32)
    if task_id ==0:
      return (x, (y, default))
    if task_id == 1:
      return (x, (default,y))

  if is_training:
    dataset = dataset.shuffle(100)
    dataset = dataset.repeat()

  dataset = dataset.map(
      _select_data_from_record,
      num_parallel_calls=tf.data.experimental.AUTOTUNE)
  #dataset = dataset.batch(batch_size, drop_remainder=is_training)
  dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
  return dataset


# Multilingual BERT configuration


config_dict = {
"attention_probs_dropout_prob": 0.1,
"directionality": "bidi",
"hidden_act": "gelu",
"hidden_dropout_prob": 0.1,
"hidden_size": 768,
"initializer_range": 0.02,
"intermediate_size": 3072,
"max_position_embeddings": 512,
"num_attention_heads": 12,
"num_hidden_layers": 12,
"pooler_fc_size": 768,
"pooler_num_attention_heads": 12,
"pooler_num_fc_layers": 3,
"pooler_size_per_head": 128,
"pooler_type": "first_token_transform",
"type_vocab_size": 2,
"vocab_size": 119547
}

bert_config = bert.configs.BertConfig.from_dict(config_dict)


# Set the path for different tasks that is to be used. By default, PAWSX is set with task_id = 0 and XNLI is set with task_id = 1


tf_records_filenames = ["gs://nts2020/xtereme/pawsx/train.en.tfrecords", "gs://nts2020/xtereme/xnli/train.en.tfrecords"]


#This snippet obtains the sampling factor of the two tasks proportional to their size.
sampling_factor = []
for fn in tf_records_filenames:
    c = 0
    for record in tf.compat.v1.python_io.tf_record_iterator(fn):
        c += 1
    sampling_factor.append(c)
    print(c)
c = sum(sampling_factor)
for i in range(0, len(sampling_factor)):
    sampling_factor[i] = sampling_factor[i]/c
sampling_factor


#The following snippet obtains the number of sentences in each of the XNLI train datasets consisting of 14 languages


tf_records_filenames = ["gs://nts2020/xtreme/translate_train/train.ar.tfrecords", "gs://nts2020/xtreme/translate_train/train.bg.tfrecords", "gs://nts2020/xtreme/translate_train/train.de.tfrecords",
                        "gs://nts2020/xtreme/translate_train/train.el.tfrecords","gs://nts2020/xtreme/translate_train/train.es.tfrecords","gs://nts2020/xtreme/translate_train/train.fr.tfrecords",
                        "gs://nts2020/xtreme/translate_train/train.hi.tfrecords","gs://nts2020/xtreme/translate_train/train.ru.tfrecords","gs://nts2020/xtreme/translate_train/train.sw.tfrecords",
                        "gs://nts2020/xtreme/translate_train/train.th.tfrecords","gs://nts2020/xtreme/translate_train/train.tr.tfrecords","gs://nts2020/xtreme/translate_train/train.ur.tfrecords",
                        "gs://nts2020/xtreme/translate_train/train.vi.tfrecords","gs://nts2020/xtreme/translate_train/train.zh.tfrecords"]


other_langs_sampling_factor = []
for fn in tf_records_filenames:
    c = 0
    for record in tf.compat.v1.python_io.tf_record_iterator(fn):
        c += 1
    other_langs_sampling_factor.append(c)
    print(c)
c = sum(other_langs_sampling_factor)
for i in range(0, len(other_langs_sampling_factor)):
    other_langs_sampling_factor[i] = other_langs_sampling_factor[i]/c
other_langs_sampling_factor


# This snippet is for controlling the sampling weights from multiple tasks manually. other_lang_aggregate_weight controls the weights given to the languages except English combined.


other_lang_count = len(tf_records_filenames)
other_lang_aggregate_weight = 0.9
train_sampling_factor = []
for i in sampling_factor:
  train_sampling_factor.append((i* (1-other_lang_aggregate_weight))/ sum(sampling_factor))
for i in other_langs_sampling_factor:
  train_sampling_factor.append((i * other_lang_aggregate_weight))




train_sampling_factor[0] = 0.03


# Uniform weighting. The first value in thhe array represents the weight for PAWSX (downweighted)


train_sampling_factor = [0.03,
 0.08882590708500053,
 0.06428571428571428,
 0.06428571428571428,
 0.06428571428571428,
 0.06428571428571428,
 0.06428571428571428,
 0.06428571428571428,
 0.06428571428571428,
 0.06428571428571428,
 0.06428571428571428,
 0.06428571428571428,
 0.06428571428571428,
 0.06428571428571428,
 0.06428571428571428,
 0.06428571428571428]


if sum(train_sampling_factor)!=1:
  train_sampling_factor[1]+= 1- sum(train_sampling_factor)


# The custom loss function below calculates loss for individual tasks based on task_id. Every data sample in a batch is encoded in the format (x, (y1, y2)) where x represents the input to the BERT model 
# while the tuple (y1, y2) represents whether the input belongs to task 1 or 2. If it belongs to task 1, it is encoded as (y1,-1) where y1 represents the label for task 1 and vice versa


def _loss_with_filter(y_true, y_pred):
  num_labels = y_pred.get_shape().as_list()[-1]
  log_probs = tf.nn.log_softmax(y_pred, axis=-1)
  log_probs = tf.reshape(log_probs, [-1, num_labels])
  labels = tf.reshape(y_true, [-1])
  one_hot_labels = tf.one_hot(labels, depth=num_labels, dtype=tf.float32)
  per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
  loss = tf.reduce_mean(per_example_loss)
  return loss


# The custom accuracy function below also calculates accuracy based on labels that are not -1s. The labels that belong to a particular tasks are compared with true labels to obtain accuracy.


import tensorflow.keras.backend as K
def accuracy_mod(y_true, y_pred):
  # Squeeze the shape to (None, ) from (None, 1) as we want to apply operations directly on y_true
  if K.ndim(y_true) == K.ndim(y_pred):
        y_true = K.squeeze(y_true, -1)

  # Normalize the y_pred values first and then take the arg at which we have a maximum value (This is the predicted label)
  y_pred = K.softmax(y_pred, axis = -1)
  y_pred = K.argmax(y_pred, axis = -1)

  # Since the ground labels can also have -1s for which we don't wanna calculate accuracy, we are filtering them off
  defa = K.constant([0], dtype=tf.float32)
  #Creating a boolean tensor for labels greater or equal to 0
  is_valid = K.greater_equal(y_true, defa)
  #Get the corresponding indices
  indices = tf.where(is_valid)

  #Gather the results of y_true and y_pred at the indices we calculated above
  fil_y_true = K.gather(y_true, K.reshape(indices, [-1])) 
  fil_y_pred = K.gather(y_pred, K.reshape(indices, [-1]))
  # K.print_tensor(res, message='res = ')
  # K.print_tensor(comp, message='comp = ')

  fil_y_true = K.cast(fil_y_true, K.floatx())
  fil_y_pred = K.cast(fil_y_pred, K.floatx())

  #pdb.set_trace()
  return K.cast(K.equal(fil_y_true, fil_y_pred), K.floatx())


# Batch size and sequence length


epochs = 3
batch_size = 64
eval_batch_size = 64
max_seq_length = 128


# Create PAWSX and XNLI EN dataset with corresponding task_ids


paws_training_dataset = create_classifier_dataset(
"gs://nts2020/xtereme/pawsx/train.en.tfrecords",
128,
batch_size,
task_id = 0,
is_training=True)


xnli_training_dataset = create_classifier_dataset(
"gs://nts2020/xtereme/xnli/train.en.tfrecords",
128,
batch_size,
task_id =1,
is_training=True)

paws_eval_dataset = create_classifier_dataset(
"gs://nts2020/xtereme/pawsx/eval.en.tfrecords",
128,
batch_size,
task_id = 0,
is_training=False)

xnli_eval_dataset = create_classifier_dataset(
"gs://nts2020/xtereme/xnli/eval.en.tfrecords",
128,
batch_size,
task_id = 1,
is_training=False)


# The training_dataset for other languages are created.


tf_records_filenames = ["gs://nts2020/xtreme/translate_train/train.ar.tfrecords", "gs://nts2020/xtreme/translate_train/train.bg.tfrecords", "gs://nts2020/xtreme/translate_train/train.de.tfrecords",
                        "gs://nts2020/xtreme/translate_train/train.el.tfrecords","gs://nts2020/xtreme/translate_train/train.es.tfrecords","gs://nts2020/xtreme/translate_train/train.fr.tfrecords",
                        "gs://nts2020/xtreme/translate_train/train.hi.tfrecords","gs://nts2020/xtreme/translate_train/train.ru.tfrecords","gs://nts2020/xtreme/translate_train/train.sw.tfrecords",
                        "gs://nts2020/xtreme/translate_train/train.th.tfrecords","gs://nts2020/xtreme/translate_train/train.tr.tfrecords","gs://nts2020/xtreme/translate_train/train.ur.tfrecords",
                        "gs://nts2020/xtreme/translate_train/train.vi.tfrecords","gs://nts2020/xtreme/translate_train/train.zh.tfrecords"]
xnli_ar_training_dataset = create_classifier_dataset(
    "gs://nts2020/xtreme/translate_train/train.ar.tfrecords",
    128,
    batch_size,
    task_id =1,
    lang_id =1,
    is_training=True)
xnli_bg_training_dataset = create_classifier_dataset(
    "gs://nts2020/xtreme/translate_train/train.bg.tfrecords",
    128,
    batch_size,
    task_id =1,
    lang_id =2,
    is_training=True)
xnli_de_training_dataset = create_classifier_dataset(
    "gs://nts2020/xtreme/translate_train/train.de.tfrecords",
    128,
    batch_size,
    task_id =1,
    lang_id =3,
    is_training=True)
xnli_el_training_dataset = create_classifier_dataset(
    "gs://nts2020/xtreme/translate_train/train.el.tfrecords",
    128,
    batch_size,
    task_id =1,
    lang_id =4,
    is_training=True)
xnli_es_training_dataset = create_classifier_dataset(
    "gs://nts2020/xtreme/translate_train/train.es.tfrecords",
    128,
    batch_size,
    task_id =1,
    lang_id =5,
    is_training=True)
xnli_fr_training_dataset = create_classifier_dataset(
    "gs://nts2020/xtreme/translate_train/train.fr.tfrecords",
    128,
    batch_size,
    task_id =1,
    lang_id =6,
    is_training=True)
xnli_hi_training_dataset = create_classifier_dataset(
    "gs://nts2020/xtreme/translate_train/train.hi.tfrecords",
    128,
    batch_size,
    task_id =1,
    lang_id =7,
    is_training=True)
xnli_ru_training_dataset = create_classifier_dataset(
    "gs://nts2020/xtreme/translate_train/train.ru.tfrecords",
    128,
    batch_size,
    task_id =1,
    lang_id =8,
    is_training=True)
xnli_sw_training_dataset = create_classifier_dataset(
    "gs://nts2020/xtreme/translate_train/train.sw.tfrecords",
    128,
    batch_size,
    task_id =1,
    lang_id =9,
    is_training=True)
xnli_th_training_dataset = create_classifier_dataset(
    "gs://nts2020/xtreme/translate_train/train.th.tfrecords",
    128,
    batch_size,
    task_id =1,
    lang_id =10,
    is_training=True)
xnli_tr_training_dataset = create_classifier_dataset(
    "gs://nts2020/xtreme/translate_train/train.tr.tfrecords",
    128,
    batch_size,
    task_id =1,
    lang_id =11,
    is_training=True)
xnli_ur_training_dataset = create_classifier_dataset(
    "gs://nts2020/xtreme/translate_train/train.ur.tfrecords",
    128,
    batch_size,
    task_id =1,
    lang_id =12,
    is_training=True)
xnli_vi_training_dataset = create_classifier_dataset(
    "gs://nts2020/xtreme/translate_train/train.vi.tfrecords",
    128,
    batch_size,
    task_id =1,
    lang_id =13,
    is_training=True)
xnli_zh_training_dataset = create_classifier_dataset(
    "gs://nts2020/xtreme/translate_train/train.zh.tfrecords",
    128,
    batch_size,
    task_id =1,
    lang_id =14,
    is_training=True)


# Eval for other XNLI languages are created.


xnli_ar_eval_dataset = create_classifier_dataset(
    "gs://nts2020/xtreme/xnli_w_dev/eval_ar.tfrecords",
    max_seq_length,
    batch_size,
    task_id =1,
    is_training=False)
xnli_bg_eval_dataset = create_classifier_dataset(
    "gs://nts2020/xtreme/xnli_w_dev/eval_bg.tfrecords",
    max_seq_length,
    batch_size,
    task_id =1,
    is_training=False)
xnli_de_eval_dataset = create_classifier_dataset(
    "gs://nts2020/xtreme/xnli_w_dev/eval_de.tfrecords",
    max_seq_length,
    batch_size,
    task_id =1,
    is_training=False)
xnli_el_eval_dataset = create_classifier_dataset(
    "gs://nts2020/xtreme/xnli_w_dev/eval_el.tfrecords",
    max_seq_length,
    batch_size,
    task_id =1,
    is_training=False)
xnli_es_eval_dataset = create_classifier_dataset(
    "gs://nts2020/xtreme/xnli_w_dev/eval_es.tfrecords",
    max_seq_length,
    batch_size,
    task_id =1,
    is_training=False)
xnli_fr_eval_dataset = create_classifier_dataset(
    "gs://nts2020/xtreme/xnli_w_dev/eval_fr.tfrecords",
    max_seq_length,
    batch_size,
    task_id =1,
    is_training=False)
xnli_hi_eval_dataset = create_classifier_dataset(
    "gs://nts2020/xtreme/xnli_w_dev/eval_hi.tfrecords",
    max_seq_length,
    batch_size,
    task_id =1,
    is_training=False)
xnli_ru_eval_dataset = create_classifier_dataset(
    "gs://nts2020/xtreme/xnli_w_dev/eval_ru.tfrecords",
    max_seq_length,
    batch_size,
    task_id =1,
    is_training=False)
xnli_sw_eval_dataset = create_classifier_dataset(
    "gs://nts2020/xtreme/xnli_w_dev/eval_sw.tfrecords",
    max_seq_length,
    batch_size,
    task_id =1,
    is_training=False)
xnli_th_eval_dataset = create_classifier_dataset(
    "gs://nts2020/xtreme/xnli_w_dev/eval_th.tfrecords",
    max_seq_length,
    batch_size,
    task_id =1,
    is_training=False)
xnli_tr_eval_dataset = create_classifier_dataset(
    "gs://nts2020/xtreme/xnli_w_dev/eval_tr.tfrecords",
    max_seq_length,
    batch_size,
    task_id =1,
    is_training=False)
xnli_ur_eval_dataset = create_classifier_dataset(
    "gs://nts2020/xtreme/xnli_w_dev/eval_ur.tfrecords",
    max_seq_length,
    batch_size,
    task_id =1,
    is_training=False)
xnli_vi_eval_dataset = create_classifier_dataset(
    "gs://nts2020/xtreme/xnli_w_dev/eval_vi.tfrecords",
    max_seq_length,
    batch_size,
    task_id =1,
    is_training=False)
xnli_zh_eval_dataset = create_classifier_dataset(
    "gs://nts2020/xtreme/xnli_w_dev/eval_zh.tfrecords",
    max_seq_length,
    batch_size,
    task_id =1,
    is_training=False)


# The training dataset with predefined sampling factor is created


training_dataset = tf.data.experimental.sample_from_datasets(
    [paws_training_dataset, xnli_training_dataset, xnli_ar_training_dataset, xnli_bg_training_dataset, xnli_de_training_dataset, xnli_el_training_dataset, xnli_es_training_dataset, 
     xnli_fr_training_dataset, xnli_hi_training_dataset, xnli_ru_training_dataset, xnli_sw_training_dataset, xnli_th_training_dataset,
     xnli_tr_training_dataset, xnli_ur_training_dataset, xnli_vi_training_dataset, xnli_zh_training_dataset], weights=tf.constant([train_sampling_factor[0], train_sampling_factor[1],
                                                                                                                                   train_sampling_factor[2], train_sampling_factor[3],
                                                                                                                                   train_sampling_factor[4],train_sampling_factor[5],
                                                                                                                                   train_sampling_factor[6],train_sampling_factor[7],
                                                                                                                                   train_sampling_factor[8],train_sampling_factor[9],
                                                                                                                                   train_sampling_factor[10],train_sampling_factor[11],
                                                                                                                                   train_sampling_factor[12],train_sampling_factor[13],
                                                                                                                                   train_sampling_factor[14],train_sampling_factor[15]]))



evaluation_sampling_factor = [0.5 , 0.5]


# Evaluation dataset consists of the target language
evaluation_dataset = tf.data.experimental.sample_from_datasets(
    [paws_eval_dataset,xnli_hi_eval_dataset], weights=tf.constant([evaluation_sampling_factor[0],evaluation_sampling_factor[1]]))                                                                                                                              


# Threshold to do early-stopping
threshold = 0.72

class myCallback(tf.keras.callbacks.Callback): 
    def on_epoch_end(self, epoch, logs={}): 
        if(logs.get('val_output1_accuracy_mod') > threshold):   
          print("\nWe have reached %2.2f%% accuracy, so we will stopping training." %(acc_thresh*100))   
          self.model.stop_training = True



callbacks = myCallback()




strategy = tf.distribute.TPUStrategy(tpu)

with strategy.scope():
    max_seq_length = 128
    initializer = tf.keras.initializers.TruncatedNormal(
            stddev=bert_config.initializer_range)
    bert_encoder = bert.bert_models.get_transformer_encoder(
        bert_config, max_seq_length)

    input_word_ids = tf.keras.layers.Input(
      shape=(max_seq_length,), dtype=tf.int32, name='input_word_ids')
    input_mask = tf.keras.layers.Input(
      shape=(max_seq_length,), dtype=tf.int32, name='input_mask')
    input_type_ids = tf.keras.layers.Input(
      shape=(max_seq_length,), dtype=tf.int32, name='input_type_ids')

    bert_model = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_multi_cased_L-12_H-768_A-12/2",
                                trainable=True)
    #bert_model = hub.KerasLayer(hub_url_bert, trainable=True)
    pooled_output, seq_output = bert_model([input_word_ids, input_mask, input_type_ids])
    output1 = tf.keras.layers.Dropout(rate=bert_config.hidden_dropout_prob)(
      pooled_output)

    output1 = tf.keras.layers.Dense(
      2, kernel_initializer=initializer, name='output1')(
          output1)

    output2 = tf.keras.layers.Dropout(rate=bert_config.hidden_dropout_prob)(
      pooled_output)

    output2 = tf.keras.layers.Dense(
      3, kernel_initializer=initializer, name='output2')(
          output2)

    model = tf.keras.Model(
          inputs={
              'input_word_ids': input_word_ids,
              'input_mask': input_mask,
              'input_type_ids': input_type_ids
          },
          outputs=[output1, output2])

    # Set up epochs and steps

    # get train_data_size from metadata
    train_data_size = c
    steps_per_epoch = int(train_data_size / batch_size)
    num_train_steps = steps_per_epoch * epochs
    warmup_steps = int(epochs * train_data_size * 0.1 / batch_size)

    # creates an optimizer with learning rate schedule
    optimizer = nlp.optimization.create_optimizer(
        2e-5, num_train_steps=num_train_steps, num_warmup_steps=warmup_steps)

  
    
    
    training_dataset = training_dataset.batch(batch_size)
    evaluation_dataset = evaluation_dataset.batch(batch_size, drop_remainder=True)
    
 

    model.compile(optimizer = optimizer, loss = [_loss_with_filter, _loss_with_filter], metrics = [accuracy_mod])
    history = model.fit(training_dataset, batch_size = batch_size, epochs= 35, steps_per_epoch = 1000, validation_data=evaluation_dataset, callbacks = [callbacks])


xnli_hi_test_dataset = create_classifier_dataset(
    "gs://nts2020/xtreme/xnli_w_dev/test_hi.tf_record",
    max_seq_length,
    batch_size,
    task_id =1,
    is_training=False)
xnli_hi_test_dataset = xnli_hi_test_dataset.batch(batch_size, drop_remainder = True)

# Calculate the test score

model.evaluate(xnli_hi_test_dataset)


from collections import defaultdict
def get_batch_lang(iterator):
  appearances = defaultdict(int)
  for curr in iterator.next()[0]['lang_id']:
    appearances[curr] += 1
    batch_lang_count = 15 *[None]
    for i in range(15):
      batch_lang_count[i] = appearances[i]
  return batch_lang_count


#Set target_lang here
import random
target_lang = 7
cce = tf.keras.losses.CategoricalCrossentropy()
d = 15

train_sampling_factor_target = [0.03,
 0.08882590708500053,
 0.06428571428571428,
 0.06428571428571428,
 0.06428571428571428,
 0.06428571428571428,
 0.06428571428571428,
 0.06428571428571428,
 0.06428571428571428,
 0.06428571428571428,
 0.06428571428571428,
 0.06428571428571428,
 0.06428571428571428,
 0.06428571428571428,
 0.06428571428571428,
 0.06428571428571428]

delta = 0.50
for i in range(0, len(train_sampling_factor)):
 train_sampling_factor_target[i] = train_sampling_factor_target[i] * (1 - delta)
train_sampling_factor_target[0] += delta

delta = 0.50
for i in range(0, len(train_sampling_factor)):
 train_sampling_factor_target[i] = train_sampling_factor_target[i] * (1 - delta)
train_sampling_factor_target[target_lang] += delta

training_dataset_target = tf.data.experimental.sample_from_datasets(
        [paws_training_dataset, xnli_training_dataset, xnli_ar_training_dataset, xnli_bg_training_dataset, xnli_de_training_dataset, xnli_el_training_dataset, xnli_es_training_dataset, 
         xnli_fr_training_dataset, xnli_hi_training_dataset, xnli_ru_training_dataset, xnli_sw_training_dataset, xnli_th_training_dataset,
         xnli_tr_training_dataset, xnli_ur_training_dataset, xnli_vi_training_dataset, xnli_zh_training_dataset], weights=tf.constant([train_sampling_factor_target[0], train_sampling_factor_target[1],
                                                                                                                                       train_sampling_factor_target[2], train_sampling_factor_target[3],
                                                                                                                                       train_sampling_factor_target[4],train_sampling_factor_target[5],
                                                                                                                                       train_sampling_factor_target[6],train_sampling_factor_target[7],
                                                                                                                                       train_sampling_factor_target[8],train_sampling_factor_target[9],
                                                                                                                                       train_sampling_factor_target[10],train_sampling_factor_target[11],
                                                                                                                                       train_sampling_factor_target[12],train_sampling_factor_target[13],
                                                                                                                                       train_sampling_factor_target[14],train_sampling_factor_target[15]]))

tgt_iterator = None
training_dataset_target = training_dataset_target.batch(32)
tgt_iterator = training_dataset_target.as_numpy_iterator()


def get_new_static_weights(reward, input1):
    epsilon = 0.05
    epsilon2 = 0.50
    acc = 0
#     initializer = tf.keras.initializers.RandomUniform(minval=0., maxval=1.)
#     input1 = tf.Variable(initializer(shape=(1, 15))[0])
    #input1 = tf.Variable(train_sampling_factor[1:])
    flag = 0
    batch_data = get_batch_lang(itera)
    for i in range(100):
        draw = random.uniform(0.0, 1.0)
        batch_data = get_batch_lang(itera)
        if draw <= epsilon:
            #print("a")
            for i in range(15):
                batch_data[i] = random.random()
            
        #if draw > epsilon and draw < epsilon2:
            #print("b")
         #   batch_data = get_batch_lang(itera)

           
        if draw >= epsilon2:
            #print("c")
            batch_data = get_batch_lang(tgt_iterator)
        input1 = trainstep(opt, input1, batch_data, reward * 100, loss)
        acc += tf.nn.softmax(input1, axis = -1)    
    return acc/100




def trainstep(opt, input1, batch_lang_count, R, loss):
   loss = 0
   with tf.GradientTape() as tape:
        #tape.watch(input1)
        for i, val in enumerate(batch_lang_count):
            loss += val * (R) * cce(tf.one_hot(i, depth =d), tf.squeeze(tf.nn.softmax(input1, axis = -1))) 
        #loss += cce(input1, tf.cast(batch_data, dtype= tf.float64))
        print(tf.nn.softmax(input1, axis = -1))

   gradients = tape.gradient(loss, input1)
   #gradients, _ = tf.clip_by_global_norm(gradients, 5.0)
   #print(gradients)
   opt.apply_gradients(zip([gradients], [input1]))
   return input1
   #print(loss)
    


Runs = 10
for run in range(Runs):
    if run!=0:
        train_sampling_factor = 15 * [None]
        train_sampling_factor[0] = 0.01
        train_sampling_factor[1:] = new_weights.numpy()
        # train_sampling_factor[1:] = [0.49758378, 0.01234279, 0.01297586, 0.01206473, 0.0130334 ,
        #        0.01228484, 0.01269346, 0.0109312 , 0.01205256, 0.01229929,
        #        0.34325856, 0.01188709, 0.01400499, 0.01077152, 0.01181594]
        # train_sampling_factor[1] -= 0.04
        if sum(train_sampling_factor)!=1:
          train_sampling_factor[1]+= 1- sum(train_sampling_factor)


    # In[322]:


    training_dataset = tf.data.experimental.sample_from_datasets(
        [paws_training_dataset, xnli_training_dataset, xnli_ar_training_dataset, xnli_bg_training_dataset, xnli_de_training_dataset, xnli_el_training_dataset, xnli_es_training_dataset, 
         xnli_fr_training_dataset, xnli_hi_training_dataset, xnli_ru_training_dataset, xnli_sw_training_dataset, xnli_th_training_dataset,
         xnli_tr_training_dataset, xnli_ur_training_dataset, xnli_vi_training_dataset, xnli_zh_training_dataset], weights=tf.constant([train_sampling_factor[0], train_sampling_factor[1],
                                                                                                                                       train_sampling_factor[2], train_sampling_factor[3],
                                                                                                                                       train_sampling_factor[4],train_sampling_factor[5],
                                                                                                                                       train_sampling_factor[6],train_sampling_factor[7],
                                                                                                                                       train_sampling_factor[8],train_sampling_factor[9],
                                                                                                                                       train_sampling_factor[10],train_sampling_factor[11],
                                                                                                                                       train_sampling_factor[12],train_sampling_factor[13],
                                                                                                                                       train_sampling_factor[14],train_sampling_factor[15]]))


    untouched_dataset = training_dataset

    training_dataset = training_dataset.batch(batch_size)

    evaluation_dataset = tf.data.experimental.sample_from_datasets(
        [paws_eval_dataset,xnli_hi_eval_dataset], weights=tf.constant([evaluation_sampling_factor[0],evaluation_sampling_factor[1]]))                                                                                                                              

    evaluation_dataset = evaluation_dataset.batch(batch_size, drop_remainder=True)

    # Refit the model. Either continue from the previous checkpoint for time-variant or retrain from scratch for time-invariant
    model.fit(training_dataset, batch_size = batch_size, epochs= 15, steps_per_epoch = 1000, validation_data=evaluation_dataset)




    itera = None
    rl_dataset = untouched_dataset#.batch(batch_size)
    rl_dataset = rl_dataset.batch(32)
    itera = rl_dataset.as_numpy_iterator()

    # Get new weights using the RL policy
    reward_accuracy = max(model.history['val_output1_accuracy_mod'])
    new_weights = get_new_static_weights(reward_accuracy, tf.Variable(train_sampling_factor[1:]))


