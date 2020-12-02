#!/usr/bin/env python
# coding: utf-8


get_ipython().system('pip install -q tf-models-official==2.3.0')



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


try:
  tpu = tf.distribute.cluster_resolver.TPUClusterResolver(tpu='tpu-quickstart')
  tf.config.experimental_connect_to_cluster(tpu)
  tf.tpu.experimental.initialize_tpu_system(tpu)# TPU detection
  print('Running on TPU ', tpu.cluster_spec().as_dict()['worker'])
except ValueError:
  raise BaseException('ERROR: Not connected to a TPU runtime; please see the previous cell in this notebook for instructions!')



tf.__version__



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
         'task_id' : task_id
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


#Change the appropriate auxiliary and target task here. Task 1 is currently set to target

tf_records_filenames = ["gs://nts2020/glue/RTE/RTE_train.tf_record", 
                        "gs://nts2020/glue/MNLI_train.tf_record"]



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




if sum(train_sampling_factor)!=1:
  train_sampling_factor[1]+= 1- sum(train_sampling_factor)




def _loss_with_filter(y_true, y_pred):
  num_labels = y_pred.get_shape().as_list()[-1]
  log_probs = tf.nn.log_softmax(y_pred, axis=-1)
  log_probs = tf.reshape(log_probs, [-1, num_labels])
  labels = tf.reshape(y_true, [-1])
  one_hot_labels = tf.one_hot(labels, depth=num_labels, dtype=tf.float32)
  per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
  loss = tf.reduce_mean(per_example_loss)
  return loss




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


#Training settings

epochs = 3
batch_size = 64
eval_batch_size = 64
max_seq_length = 128


#Pass appropriate tf records to get the training and eval data for GLUE tasks

rte_training_dataset = create_classifier_dataset(
"gs://nts2020/glue/RTE/RTE_train.tf_record",
128,
batch_size,
task_id = 0,
is_training=True)


qnli_training_dataset = create_classifier_dataset(
"gs://nts2020/glue/QNLI/QNLI_train.tf_record",
128,
batch_size,
task_id =1,
is_training=True)

mnli_training_dataset = create_classifier_dataset(
"gs://nts2020/glue/MNLI_train.tf_record",
128,
batch_size,
task_id =1,
is_training=True)

wnli_training_dataset = create_classifier_dataset(
"gs://nts2020/glue/WNLI/WNLI_train.tf_record",
128,
batch_size,
task_id =1,
is_training=True)

mrpc_training_dataset = create_classifier_dataset(
"gs://nts2020/glue/MRPC/MRPC_train.tf_record",
128,
batch_size,
task_id =0,
is_training=True)

rte_eval_dataset = create_classifier_dataset(
"gs://nts2020/glue/RTE/RTE_eval.tf_record",
128,
batch_size,
task_id = 0,
is_training=False)

qnli_eval_dataset = create_classifier_dataset(
"gs://nts2020/glue/QNLI/QNLI_eval.tf_record",
128,
batch_size,
task_id = 1,
is_training=False)

mnli_eval_dataset = create_classifier_dataset(
"gs://nts2020/glue/MNLI_eval.tf_record",
128,
batch_size,
task_id =1,
is_training=False)

wnli_eval_dataset = create_classifier_dataset(
"gs://nts2020/glue/WNLI/WNLI_eval.tf_record",
128,
batch_size,
task_id =1,
is_training=False)

mrpc_eval_dataset = create_classifier_dataset(
"gs://nts2020/glue/MRPC/MRPC_eval.tf_record",
128,
batch_size,
task_id =0,
is_training=False)







#Training sampling factpr

train_sampling_factor = [0.1 , 0.9]


# Create a training dataset based on the above-mentioned sampling factor


training_dataset = tf.data.experimental.sample_from_datasets(
     [rte_training_dataset, mnli_training_dataset], weights = tf.constant([train_sampling_factor[0] , train_sampling_factor[1]]))


# iterator = training_dataset.as_numpy_iterator()
# iterator.next()


# training_dataset = tf.data.experimental.sample_from_datasets(
#     [paws_training_dataset, xnli_sw_training_dataset], weights = tf.constant([0.5,0.5]))


#training_dataset = rte_training_dataset

evaluation_sampling_factor = [0.5, 0.5]




evaluation_dataset = tf.data.experimental.sample_from_datasets(
    [rte_eval_dataset,mnli_eval_dataset], weights=tf.constant([evaluation_sampling_factor[0],evaluation_sampling_factor[1]]))                                                                                                                              



# evaluation_dataset = tf.data.experimental.sample_from_datasets(
#     [paws_eval_dataset, xnli_eval_dataset], weights=tf.constant([sampling_factor[0], sampling_factor[1]])
# )


## Set accuracy_threshold to do early-stopping or save every checkpoint and pick the best on eval

acc_thresh = 0.92



class myCallback(tf.keras.callbacks.Callback): 
    def on_epoch_end(self, epoch, logs={}): 
        if(logs.get('val_output1_accuracy_mod') > acc_thresh):   
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

    bert_model = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/2",
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
    history = model.fit(training_dataset, batch_size = batch_size, epochs= 7, steps_per_epoch = 1000, validation_data=evaluation_dataset, callbacks=[callbacks])




model.fit(training_dataset, batch_size = batch_size, epochs= 3, steps_per_epoch = 1000, validation_data=evaluation_dataset, callbacks=[callbacks])



