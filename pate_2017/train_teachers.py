# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from models import deep_cnn, rf, xgb, cat, svm, logistic
import input  # pylint: disable=redefined-builtin
import metrics
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()
tf.compat.v1.disable_eager_execution()


tf.flags.DEFINE_string('dataset', 'svhn', 'The name of the dataset to use')
tf.flags.DEFINE_integer('nb_labels', 10, 'Number of output classes')
tf.flags.DEFINE_string('model', 'cnn', 'The name of the model to use')

tf.flags.DEFINE_string('data_dir',f'/home/aix21202/pate/privacy/research/pate_2017/tmp','Temporary storage')
tf.flags.DEFINE_string('train_dir',f'/home/aix21202/pate/privacy/research/pate_2017/tmp/train_dir',
                       'Where model ckpt are saved')

tf.flags.DEFINE_integer('max_steps', 3000, 'Number of training steps to run.')
tf.flags.DEFINE_integer('nb_teachers', 50, 'Teachers in the ensemble.')
tf.flags.DEFINE_integer('teacher_id', 0, 'ID of teacher being trained.')

tf.flags.DEFINE_boolean('deeper', False, 'Activate deeper CNN model')

FLAGS = tf.flags.FLAGS

print(f"deep: {FLAGS.deeper}")

def train_teacher(dataset, nb_teachers, teacher_id, model):
  """
  This function trains a teacher (teacher id) among an ensemble of nb_teachers
  models for the dataset specified.
  :param dataset: string corresponding to dataset (svhn, cifar10)
  :param nb_teachers: total number of teachers in the ensemble
  :param teacher_id: id of the teacher being trained
  :return: True if everything went well
  """
  # If working directories do not exist, create them
  assert input.create_dir_if_needed(FLAGS.data_dir)
  assert input.create_dir_if_needed(FLAGS.train_dir)

  # Load the dataset
  if dataset == 'svhn':
    train_data,train_labels,test_data,test_labels = input.ld_svhn(extended=True)
  elif dataset == 'cifar10':
    train_data, train_labels, test_data, test_labels = input.ld_cifar10()
  elif dataset == 'mnist':
    train_data, train_labels, test_data, test_labels = input.ld_mnist()
  elif dataset == 'remosa':
        train_data, train_labels, test_data, test_labels = input.ld_eumc_remosa()
  elif dataset == 'posa':
        train_data, train_labels, test_data, test_labels = input.ld_eumc_posa()
  else:
    print("Check value of dataset flag")
    return False

  # Retrieve subset of data for this teacher
  data, labels = input.partition_dataset(train_data,
                                         train_labels,
                                         nb_teachers,
                                         teacher_id)

  print("Length of training data: " + str(len(labels)))
  
  if FLAGS.deeper:
    filename = filename = f"{nb_teachers}_teachers_{teacher_id}_deep_cnn.ckpt"
  else:
    if FLAGS.model:
      filename = f"{nb_teachers}_teachers_{teacher_id}_{model}.ckpt"
    else:
      filename = f"{nb_teachers}_teachers_{teacher_id}.ckpt"
  ckpt_path = f'{FLAGS.train_dir}/{dataset}_{filename}'
  
  print(f'Saving checkpoint to: {ckpt_path}')  # Print checkpoint path for verification
  print(f"model: {model}")
  
  if model == 'cnn':
    # Perform teacher training
    assert deep_cnn.train(data, labels, ckpt_path)

    # Append final step value to checkpoint for evaluation
    ckpt_path_final = ckpt_path + '-' + str(FLAGS.max_steps - 1)

    # Retrieve teacher probability estimates on the test data
    teacher_preds = deep_cnn.softmax_preds(test_data, ckpt_path_final)

    # Compute teacher accuracy
    precision = metrics.accuracy(teacher_preds, test_labels)
    print('Precision of teacher after training: ' + str(precision))

    return True
  
  elif model == 'rf':
    assert rf.train(data, labels, ckpt_path)
    
    # Load the model checkpoint for evaluation
    ckpt_path_final = ckpt_path

    # Retrieve teacher probability estimates on the test data
    teacher_preds = rf.predict(test_data, ckpt_path_final)

    # Compute teacher accuracy using the Random Forest model
    precision = rf.evaluate(test_data, test_labels, ckpt_path_final)
    print('Precision of Random Forest teacher after training: ' + str(precision))

    return True
  
  elif model == 'xgb':
    assert xgb.train(data, labels, ckpt_path)
    
    # Load the model checkpoint for evaluation
    ckpt_path_final = ckpt_path

    # Retrieve teacher probability estimates on the test data
    teacher_preds = xgb.predict(test_data, ckpt_path_final)

    # Compute teacher accuracy using the Random Forest model
    precision = xgb.evaluate(test_data, test_labels, ckpt_path_final)
    # print('Precision of XGBoost teacher after training: ' + str(precision))

    return True
    # return precision
  
  elif model == 'cat':
    assert cat.train(data, labels, ckpt_path)
    
    # Load the model checkpoint for evaluation
    ckpt_path_final = ckpt_path

    # Retrieve teacher probability estimates on the test data
    teacher_preds = cat.predict(test_data, ckpt_path_final)

    # Compute teacher accuracy using the Random Forest model
    precision = cat.evaluate(test_data, test_labels, ckpt_path_final)
    print('Precision of CatBoost teacher after training: ' + str(precision))

    return True
  
  elif model == 'svm':
    assert svm.train(data, labels, ckpt_path)
    
    # Load the model checkpoint for evaluation
    ckpt_path_final = ckpt_path

    # Retrieve teacher probability estimates on the test data
    teacher_preds = svm.predict(test_data, ckpt_path_final)

    # Compute teacher accuracy using the Random Forest model
    precision = svm.evaluate(test_data, test_labels, ckpt_path_final)
    print('Precision of Support Vector Machine teacher after training: ' + str(precision))

    return True
  
  elif model == 'logistic':
    assert logistic.train(data, labels, ckpt_path)
    
    # Load the model checkpoint for evaluation
    ckpt_path_final = ckpt_path

    # Retrieve teacher probability estimates on the test data
    teacher_preds = logistic.predict(test_data, ckpt_path_final)

    # Compute teacher accuracy using the Random Forest model
    precision = logistic.evaluate(test_data, test_labels, ckpt_path_final)
    # print('Precision of Support Vector Machine teacher after training: ' + str(precision))
    
    return True
  


def main(argv=None):  # pylint: disable=unused-argument
  # Make a call to train_teachers with values specified in flags
  assert train_teacher(FLAGS.dataset, FLAGS.nb_teachers, FLAGS.teacher_id, FLAGS.model)

if __name__ == '__main__':
  tf.app.run()
