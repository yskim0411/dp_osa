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
import aggregation
from models import deep_cnn, rf, xgb, cat, svm, logistic
import input  # pylint: disable=redefined-builtin
import metrics
import numpy as np
from six.moves import xrange
import tensorflow.compat.v1 as tf
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold 
import joblib
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score

tf.disable_v2_behavior()
tf.compat.v1.disable_eager_execution()

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string('dataset', 'svhn', 'The name of the dataset to use')
tf.flags.DEFINE_integer('nb_labels', 10, 'Number of output classes')
tf.flags.DEFINE_string('model', 'cnn', 'The name of the model to use')

tf.flags.DEFINE_string('data_dir',f'/home/aix21202/pate/privacy/research/pate_2017/tmp','Temporary storage')
tf.flags.DEFINE_string('train_dir',f'/home/aix21202/pate/privacy/research/pate_2017/tmp/train_dir','Where model ckpt are saved')
tf.flags.DEFINE_string('teachers_dir',f'/home/aix21202/pate/privacy/research/pate_2017/tmp/train_dir',
                       'Directory wh ere teachers checkpoints are stored.')

tf.flags.DEFINE_integer('teachers_max_steps', 3000,
                        'Number of steps teachers were ran.')
tf.flags.DEFINE_integer('max_steps', 3000, 'Number of steps to run student.')
tf.flags.DEFINE_integer('nb_teachers', 10, 'Teachers in the ensemble.')
tf.flags.DEFINE_integer('stdnt_share', 1000,
                        'Student share (last index) of the test data')
tf.flags.DEFINE_integer('lap_scale', 10,
                        'Scale of the Laplacian noise added for privacy')
tf.flags.DEFINE_boolean('save_labels', False,
                        'Dump numpy arrays of labels and clean teacher votes')
tf.flags.DEFINE_boolean('deeper', False, 'Activate deeper CNN model')


def ensemble_preds(dataset, nb_teachers, stdnt_data, model_name):
  """
  Given a dataset, a number of teachers, and some input data, this helper
  function queries each teacher for predictions on the data and returns
  all predictions in a single array. (That can then be aggregated into
  one single prediction per input using aggregation.py (cf. function
  prepare_student_data() below)
  :param dataset: string corresponding to mnist, cifar10, or svhn
  :param nb_teachers: number of teachers (in the ensemble) to learn from
  :param stdnt_data: unlabeled student training data
  :return: 3d array (teacher id, sample id, probability per class)
  """
  
  # 모델 이름에 따라 실제 모델 함수/클래스를 매핑
  model_map = {
        'deep_cnn': deep_cnn,
        'rf': rf,
        'xgb': xgb,
        'cat': cat,
        'svm': svm,
        'logistic': logistic
    }
  # model_name에 맞는 모델 객체를 가져옴
  model = model_map.get(model_name)

  # Compute shape of array that will hold probabilities produced by each
  # teacher, for each training point, and each output class
  result_shape = (nb_teachers, len(stdnt_data), FLAGS.nb_labels)

  # Create array that will hold result
  result = np.zeros(result_shape, dtype=np.float32)

  # Get predictions from each teacher
  for teacher_id in xrange(nb_teachers):
    # Compute path of checkpoint file for teacher model with ID teacher_id
    if FLAGS.deeper:
      ckpt_path = FLAGS.teachers_dir + '/' + str(dataset) + '_' + str(nb_teachers) + '_teachers_' + str(teacher_id) + '_deep_cnn.ckpt-' + str(FLAGS.teachers_max_steps - 1) #NOLINT(long-line)
    else:
      if model == 'deep_cnn':
        ckpt_path = FLAGS.teachers_dir + '/' + str(dataset) + '_' + str(nb_teachers) + '_teachers_' + str(teacher_id) + f'_{model_name}.ckpt-' + str(FLAGS.teachers_max_steps - 1)  # NOLINT(long-line)
      else:
        ckpt_path = FLAGS.teachers_dir + '/' + str(dataset) + '_' + str(nb_teachers) + '_teachers_' + str(teacher_id) + f'_{model_name}.ckpt'  # NOLINT(long-line)
    # print(f"model_name: {model_name}")
    if model_name == 'deep_cnn':
      # Get predictions on our training data and store in result array
      result[teacher_id] = model.softmax_preds(stdnt_data, ckpt_path)
    else:
      result[teacher_id] = model.predict(stdnt_data, ckpt_path)

    # This can take a while when there are a lot of teachers so output status
    print("Computed Teacher " + str(teacher_id) + " softmax predictions")

  return result


def prepare_student_data(dataset, nb_teachers, save=False):
  """
  Takes a dataset name and the size of the teacher ensemble and prepares
  training data for the student model, according to parameters indicated
  in flags above.
  :param dataset: string corresponding to mnist, cifar10, or svhn
  :param nb_teachers: number of teachers (in the ensemble) to learn from
  :param save: if set to True, will dump student training labels predicted by
               the ensemble of teachers (with Laplacian noise) as npy files.
               It also dumps the clean votes for each class (without noise) and
               the labels assigned by teachers
  :return: pairs of (data, labels) to be used for student training and testing
  """
  assert input.create_dir_if_needed(FLAGS.train_dir)

  # Load the dataset
  if dataset == 'svhn':
    test_data, test_labels = input.ld_svhn(test_only=True)
  elif dataset == 'cifar10':
    test_data, test_labels = input.ld_cifar10(test_only=True)
  elif dataset == 'mnist':
    test_data, test_labels = input.ld_mnist(test_only=True)
  elif dataset == 'posa':
    test_data, test_labels = input.ld_eumc_posa(test_only=True)
  elif dataset == 'remosa':
    test_data, test_labels = input.ld_eumc_remosa(test_only=True)
  elif dataset == 'remosa_tab':
    test_data, test_labels, synthetic_data, synthetic_labels = input.ld_eumc_remosa_tab(test_only=True)
  elif dataset == 'posa_tab':
    test_data, test_labels, synthetic_data, synthetic_labels = input.ld_eumc_posa_tab(test_only=True)
  else:
    print("Check value of dataset flag")
    return False

  # Make sure there is data leftover to be used as a test set
  # assert FLAGS.stdnt_share < len(test_data)
  assert FLAGS.stdnt_share <= len(synthetic_data)

  # Prepare [unlabeled] student training data (subset of test set)
  # stdnt_data = test_data[:FLAGS.stdnt_share]
  stdnt_data = synthetic_data[:FLAGS.stdnt_share]

  # Compute teacher predictions for student training data
  teachers_preds = ensemble_preds(dataset, nb_teachers, stdnt_data, FLAGS.model)


  # Aggregate teacher predictions to get student training labels
  if not save:
    stdnt_labels = aggregation.noisy_max(teachers_preds, FLAGS.lap_scale, FLAGS.nb_labels)
  else:
    # Request clean votes and clean labels as well
    stdnt_labels, clean_votes, labels_for_dump = aggregation.noisy_max(teachers_preds, FLAGS.lap_scale, FLAGS.nb_labels, return_clean_votes=True) #NOLINT(long-line)
    print(f"clean_votes shape: {clean_votes.shape}")
    print(f"clean_votes: {clean_votes}")
    # Prepare filepath for numpy dump of clean votes
    filepath = FLAGS.data_dir + "/" + str(dataset) + '_' + str(nb_teachers) + '_student_clean_votes_lap_' + str(FLAGS.lap_scale) + '.npy'  # NOLINT(long-line)

    # Prepare filepath for numpy dump of clean labels
    filepath_labels = FLAGS.data_dir + "/" + str(dataset) + '_' + str(nb_teachers) + '_teachers_labels_lap_' + str(FLAGS.lap_scale) + '.npy'  # NOLINT(long-line)

    # Dump clean_votes array
    with tf.gfile.Open(filepath, mode='w') as file_obj:
      np.save(file_obj, clean_votes)

    # Dump labels_for_dump array
    with tf.gfile.Open(filepath_labels, mode='w') as file_obj:
      np.save(file_obj, labels_for_dump)

  # Print accuracy of aggregated labels
  # ac_ag_labels = metrics.accuracy(stdnt_labels, test_labels[:FLAGS.stdnt_share])
  ac_ag_labels = metrics.accuracy(stdnt_labels, synthetic_labels[:FLAGS.stdnt_share])
  print("Accuracy of the aggregated labels: " + str(ac_ag_labels))
  
  # Store unused part of test set for use as a test set after student training
  # stdnt_test_data = test_data[FLAGS.stdnt_share:]
  # stdnt_test_labels = test_labels[FLAGS.stdnt_share:]
  stdnt_test_data = test_data
  stdnt_test_labels = test_labels
  
  if save:
    # Prepare filepath for numpy dump of labels produced by noisy aggregation
    filepath = FLAGS.data_dir + "/" + str(dataset) + '_' + str(nb_teachers) + '_student_labels_lap_' + str(FLAGS.lap_scale) + '.npy' #NOLINT(long-line)

    # Dump student noisy labels array
    with tf.gfile.Open(filepath, mode='w') as file_obj:
      np.save(file_obj, stdnt_labels)

  return stdnt_data, stdnt_labels, stdnt_test_data, stdnt_test_labels


def train_student(dataset, nb_teachers, model_name):
  """
  This function trains a student using predictions made by an ensemble of
  teachers. The student and teacher models are trained using the same
  neural network architecture.
  :param dataset: string corresponding to mnist, cifar10, or svhn
  :param nb_teachers: number of teachers (in the ensemble) to learn from
  :return: True if student training went well
  """
  assert input.create_dir_if_needed(FLAGS.train_dir)
  
  # 모델 이름을 실제 모델 객체로 매핑
  model_map = {
        'deep_cnn': deep_cnn,
        'rf': rf,
        'xgb': xgb,
        'cat': cat,
        'svm': svm,
        'logistic': logistic
    }

  model = model_map.get(model_name)

  # Call helper function to prepare student data using teacher predictions
  stdnt_dataset = prepare_student_data(dataset, nb_teachers, save=True)

  # Unpack the student dataset
  stdnt_data, stdnt_labels, stdnt_test_data, stdnt_test_labels = stdnt_dataset

  # Prepare checkpoint filename and path
  if FLAGS.deeper:
    ckpt_path = FLAGS.train_dir + '/' + str(dataset) + '_' + str(nb_teachers) + '_student_deeper.ckpt' #NOLINT(long-line)
  else:
    if model == 'deep_cnn':
      ckpt_path = FLAGS.train_dir + '/' + str(dataset) + '_' + str(nb_teachers) + '_student.ckpt'  # NOLINT(long-line)
    else:
      ckpt_path = FLAGS.train_dir + '/' + str(dataset) + '_' + str(nb_teachers) + f'_student_{model_name}.ckpt'  # NOLINT(long-line)
      model.train(stdnt_data, stdnt_labels, ckpt_path)

  # Start student training
  # assert deep_cnn.train(stdnt_data, stdnt_labels, ckpt_path)
  assert model.train(stdnt_data, stdnt_labels, ckpt_path)

  # print(f"model: {model}")
  # Compute final checkpoint name for student (with max number of steps)
  # if model == 'deep_cnn':
  if FLAGS.deeper:
    ckpt_path_final = ckpt_path + '-' + str(FLAGS.max_steps - 1)
    
    # Compute student label predictions on remaining chunk of test set
    student_preds = model.softmax_preds(stdnt_test_data, ckpt_path_final)
    
    # Compute teacher accuracy
    precision = metrics.accuracy(student_preds, stdnt_test_labels)
    print('Precision of student after training: ' + str(precision))

  else:
    ckpt_path_final = ckpt_path
    model.evaluate(stdnt_test_data, stdnt_test_labels, ckpt_path_final)
    
  return True


def cross_validate_student(dataset, nb_teachers, model_name, k_folds=5):
    """
    This function applies k-fold cross-validation to the student training,
    then trains a final model on the whole dataset using the best condition found based on AUROC.
    If AUROC is tied, selects the model with the higher accuracy.
    """
    # 모델 이름을 실제 모델 객체로 매핑
    model_map = {
        'deep_cnn': deep_cnn,
        'rf': rf,
        'xgb': xgb,
        'cat': cat,
        'svm': svm,
        'logistic': logistic
    }
    model = model_map.get(model_name)
    
    # 데이터셋 로드
    stdnt_data, stdnt_labels, stdnt_test_data, stdnt_test_labels = prepare_student_data(dataset, nb_teachers, save=True)
    
    # KFold 설정
    kfold = KFold(n_splits=k_folds, shuffle=True, random_state=40)
    best_auroc = -1
    best_accuracy = -1
    best_model_path = None
    fold_acc = []
    fold_auroc = []
    fold_f1 = []
    
    print(f"stndt_labels: {stdnt_labels}")
    # K-fold cross-validation 루프
    for fold, (train_idx, val_idx) in enumerate(kfold.split(stdnt_data)):
        print(f'Fold {fold+1}/{k_folds}')
        
        # 학습 및 검증 데이터 분리
        train_data, val_data = stdnt_data.iloc[train_idx], stdnt_data.iloc[val_idx]
        train_labels, val_labels = stdnt_labels[train_idx], stdnt_labels[val_idx]
        print(f"train_labels: {train_labels}")
        print(f"val_labels: {val_labels}")
        
        # Fold 내 데이터 개수 출력
        print(f'Number of training samples in Fold {fold+1}: {len(train_data)}')
        print(f'Number of validation samples in Fold {fold+1}: {len(val_data)}')

        # 체크포인트 경로 설정
        ckpt_path = FLAGS.train_dir + f'/{dataset}_{nb_teachers}_student_{model_name}_fold{fold}.ckpt'

        # 각 폴드에 대해 모델 학습
        model.train(train_data, train_labels, ckpt_path)
        
        # 폴드별 모델 평가
        accuracy, auroc, f1 = model.evaluate(val_data, val_labels, ckpt_path)

        # AUROC 및 Accuracy 출력
        print(f'Model accuracy for fold {fold+1}: {accuracy * 100:.2f}%')
        print(f'Model AUROC for fold {fold+1}: {auroc:.2f}')

        fold_acc.append(accuracy)
        if auroc is not None:
            fold_auroc.append(auroc)

        # 가장 좋은 AUROC이거나, AUROC가 같을 경우 accuracy가 더 높은 모델을 저장
        if auroc is not None:
            if (auroc > best_auroc) or (auroc == best_auroc and accuracy > best_accuracy):
                best_auroc = auroc
                best_accuracy = accuracy
                best_model_path = f'best_model_fold_{fold}.pkl'
                joblib.dump(model, best_model_path)  # 모델을 파일로 저장

    # 평균 정확도 및 AUROC 계산
    if fold_acc:
        avg_acc = np.mean(fold_acc)
        print(f'Average accuracy across {k_folds} folds: {avg_acc}')
    else:
        print('No valid accuracy values were computed across the folds.')
    
    if fold_auroc:
        avg_auroc = np.mean(fold_auroc)
        print(f'Average AUROC across {k_folds} folds: {avg_auroc}')
    else:
        print('No valid AUROC values were computed across the folds.')

    # 최적 조건을 사용하여 전체 데이터로 모델 훈련
    final_ckpt_path = FLAGS.train_dir + f'/{dataset}_{nb_teachers}_final_student_{model_name}.ckpt'
    if best_model_path is not None:
        print(f'Using the best model with AUROC: {best_auroc} and accuracy: {best_accuracy}')
        model = joblib.load(best_model_path)  # 최적의 모델을 불러옴
    print(f"stdnt_labels:{stdnt_labels}")
    model.train(stdnt_data, stdnt_labels, final_ckpt_path)

    print("===============================================")
    # 최종 모델 평가 - accuracy와 AUROC 모두 사용
    final_accuracy, final_auroc, final_f1 = model.evaluate(stdnt_test_data, stdnt_test_labels, final_ckpt_path)

    print('Final accuracy of the student after training on full data: ' + str(final_accuracy))
    print('Final AUROC of the student after training on full data: ' + str(final_auroc))
    print('Final f1 of the student after training on full data: ' + str(final_f1))

    return True



def main(argv=None): # pylint: disable=unused-argument
  # Run student training according to values specified in flags
  # assert train_student(FLAGS.dataset, FLAGS.nb_teachers, FLAGS.model)
  # 5-fold cross-validation 수행
  assert cross_validate_student(FLAGS.dataset, FLAGS.nb_teachers, FLAGS.model, k_folds=5)


if __name__ == '__main__':
  tf.app.run()
