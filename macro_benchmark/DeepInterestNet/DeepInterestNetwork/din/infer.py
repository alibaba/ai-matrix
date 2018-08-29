import os
import time
import pickle
import random
import numpy as np
import tensorflow as tf
import sys
from input import DataInput, DataInputTest
from model import Model

random.seed(1234)
np.random.seed(1234)
tf.set_random_seed(1234)


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
def _eval(sess, model):
  auc_sum = 0.0
  score_arr = []

  test_batch_size = [256, 512, 1024]
  print('Round    Batch size   Recommendations / sec')
  for batch in test_batch_size:
    iteration = 0
    # warm up
    for _, uij in DataInputTest(test_set, batch):
        auc_, score_ = model.eval(sess, uij)
        iteration += 1
        if iteration == 5:
            break

    perf_total = []
    for i in range(5):
        time_st = time.time()
        iteration = 0
        for _, uij in DataInputTest(test_set, batch):
            auc_, score_ = model.eval(sess, uij)
            iteration += 1
        #score_arr += _auc_arr(score_)
        #auc_sum += auc_ * len(uij[0])
    #test_gauc = auc_sum / len(test_set)
    #Auc = calc_auc(score_arr)
        time_dur = time.time() - time_st
        perf = batch * iteration / time_dur
        perf_total.append(perf)
        print(' %2i      %4i        %10.1f' % (i, batch, perf))
    print('Average performance is %10.1f for batch size=' % (sum(perf_total)/float(len(perf_total))), batch)


gpu_options = tf.GPUOptions(allow_growth=True)
with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:

  model = Model(user_count, item_count, cate_count, cate_list)
  model.restore(sess, 'save_path/ckpt') 
  start_time = time.time()
  _eval(sess, model)
