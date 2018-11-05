import os
import time
import pickle
import random
import numpy as np
import tensorflow as tf
import sys
from input import DataInput, DataInputTest
from model import Model
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", type=int, default=256, help="inference batch size")
args = parser.parse_args()

random.seed(1234)
np.random.seed(1234)
tf.set_random_seed(1234)

predict_batch_size = args.batch_size
predict_ads_num = 100


with open('dataset.pkl', 'rb') as f:
  train_set = pickle.load(f)
  test_set = pickle.load(f)
  cate_list = pickle.load(f)
  user_count, item_count, cate_count = pickle.load(f)

best_auc = 0.0

def _auc_arr(score):
  score_p = score[:,0]
  score_n = score[:,1]
  #print "============== p ============="
  #print score_p
  #print "============== n ============="
  #print score_n
  score_arr = []
  for s in score_p.tolist():
    score_arr.append([0, 1, s])
  for s in score_n.tolist():
    score_arr.append([1, 0, s])
  return score_arr


def _test(sess, model):
  
  print('Round    Batch size   Recommendations / sec')
  total_time = 0
  perf_total = []
  score_append = np.empty((predict_batch_size, predict_ads_num, 1), float)
  iteration = 0
  # warp up
  for _, uij in DataInputTest(test_set, predict_batch_size):
    score_ = model.test(sess,uij)
    score_append = np.append(score_append, score_, axis = 0)
    iteration += 1
    if iteration == 5:
      np.save('inference_' + str(predict_batch_size) +'.npy', score_append)
      break
  # start testing
  time_st = time.time()
  iteration = 0
  for _, uij in DataInputTest(test_set, predict_batch_size):
    s_time = time.time()
    score_ = model.test(sess, uij)
    e_time = time.time()
    total_time += e_time - s_time

    iteration += 1
    if iteration == 1000:
      time_dur = time.time() - time_st
      perf = predict_batch_size * iteration * predict_ads_num / time_dur
      print(' %2i      %4i        %10.1f' % (iteration, predict_batch_size, perf ))
      break
    elif iteration % 100 == 0:
      time_dur = time.time() - time_st
      perf = predict_batch_size * iteration * predict_ads_num / time_dur
      print(' %2i      %4i        %10.1f' % (iteration, predict_batch_size, perf ))
  time_dur = time.time() - time_st
  perf = predict_batch_size * iteration * predict_ads_num / time_dur
  print("Average performance is %10.1f for batch size=" % perf, predict_batch_size)
  print("Approximate accelerator time in seconds: %.3f" % total_time)


gpu_options = tf.GPUOptions(allow_growth=True)
with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
  model = Model(user_count, item_count, cate_count, cate_list, predict_batch_size, predict_ads_num)
  model.restore(sess, 'save_path/ckpt') 
  _test(sess, model)
