import numpy
from data_iterator import DataIterator
import tensorflow as tf
from model import *
import time
import random
import sys
from utils import *

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--mode", type=str, default='train', help="mode, train or test")
parser.add_argument("--model", type=str, default='DIEN', help="model")
parser.add_argument("--seed", type=int, default=3, help="seed value")
parser.add_argument("--batch_size", type=int, default=128, help="batch size")
parser.add_argument("--data_type", type=str, default='FP32', help="data type: FP32 or FP16")
parser.add_argument("--num_accelerators", type=int, default=1, help="number of accelerators used for training")
parser.add_argument("--embedding_device", type=str, default='gpu', help="synthetic input embedding layer reside on gpu or cpu")
args = parser.parse_args()

EMBEDDING_DIM = 18
HIDDEN_SIZE = 18 * 2
ATTENTION_SIZE = 18 * 2
best_auc = 0.0

TOTAL_TRAIN_SIZE = 512000
#TOTAL_TRAIN_SIZE = 16000


def prepare_data(input, target, maxlen = None, return_neg = False):
    # x: a list of sentences
    lengths_x = [len(s[4]) for s in input]
    seqs_mid = [inp[3] for inp in input]
    seqs_cat = [inp[4] for inp in input]
    noclk_seqs_mid = [inp[5] for inp in input]
    noclk_seqs_cat = [inp[6] for inp in input]
    if maxlen is not None:
        new_seqs_mid = []
        new_seqs_cat = []
        new_noclk_seqs_mid = []
        new_noclk_seqs_cat = []
        new_lengths_x = []
        for l_x, inp in zip(lengths_x, input):
            if l_x > maxlen:
                new_seqs_mid.append(inp[3][l_x - maxlen:])
                new_seqs_cat.append(inp[4][l_x - maxlen:])
                new_noclk_seqs_mid.append(inp[5][l_x - maxlen:])
                new_noclk_seqs_cat.append(inp[6][l_x - maxlen:])
                new_lengths_x.append(maxlen)
            else:
                new_seqs_mid.append(inp[3])
                new_seqs_cat.append(inp[4])
                new_noclk_seqs_mid.append(inp[5])
                new_noclk_seqs_cat.append(inp[6])
                new_lengths_x.append(l_x)
        lengths_x = new_lengths_x
        seqs_mid = new_seqs_mid
        seqs_cat = new_seqs_cat
        noclk_seqs_mid = new_noclk_seqs_mid
        noclk_seqs_cat = new_noclk_seqs_cat

        if len(lengths_x) < 1:
            return None, None, None, None
    
    n_samples = len(seqs_mid)
    maxlen_x = numpy.max(lengths_x)
    neg_samples = len(noclk_seqs_mid[0][0])


    mid_his = numpy.zeros((n_samples, maxlen_x)).astype('int64')
    cat_his = numpy.zeros((n_samples, maxlen_x)).astype('int64')
    noclk_mid_his = numpy.zeros((n_samples, maxlen_x, neg_samples)).astype('int64')
    noclk_cat_his = numpy.zeros((n_samples, maxlen_x, neg_samples)).astype('int64')
    if args.data_type == 'FP32':
        data_type = 'float32'
    elif args.data_type == 'FP16':
        data_type = 'float16'
    else:
        raise ValueError("Invalid model data type: %s" % args.data_type)
    mid_mask = numpy.zeros((n_samples, maxlen_x)).astype(data_type)
    for idx, [s_x, s_y, no_sx, no_sy] in enumerate(zip(seqs_mid, seqs_cat, noclk_seqs_mid, noclk_seqs_cat)):
        mid_mask[idx, :lengths_x[idx]] = 1.
        mid_his[idx, :lengths_x[idx]] = s_x
        cat_his[idx, :lengths_x[idx]] = s_y
        noclk_mid_his[idx, :lengths_x[idx], :] = no_sx
        noclk_cat_his[idx, :lengths_x[idx], :] = no_sy

    uids = numpy.array([inp[0] for inp in input])
    mids = numpy.array([inp[1] for inp in input])
    cats = numpy.array([inp[2] for inp in input])

    if return_neg:
        return uids, mids, cats, mid_his, cat_his, mid_mask, numpy.array(target), numpy.array(lengths_x), noclk_mid_his, noclk_cat_his

    else:
        return uids, mids, cats, mid_his, cat_his, mid_mask, numpy.array(target), numpy.array(lengths_x)

def eval(sess, test_data, model, model_path):
    loss_sum = 0.
    accuracy_sum = 0.
    aux_loss_sum = 0.
    nums = 0
    stored_arr = []
    eval_time = 0
    for src, tgt in test_data:
        nums += 1
        sys.stdout.flush()
        uids, mids, cats, mid_his, cat_his, mid_mask, target, sl, noclk_mids, noclk_cats = prepare_data(src, tgt, return_neg=True)
        # print("begin evaluation")
        start_time = time.time()
        prob, loss, acc, aux_loss = model.calculate(sess, [uids, mids, cats, mid_his, cat_his, mid_mask, target, sl, noclk_mids, noclk_cats])
        end_time = time.time()
        # print("evaluation time of one batch: %.3f" % (end_time - start_time))
        # print("end evaluation")
        eval_time += end_time - start_time
        loss_sum += loss
        aux_loss_sum = aux_loss
        accuracy_sum += acc
        prob_1 = prob[:, 0].tolist()
        target_1 = target[:, 0].tolist()
        for p ,t in zip(prob_1, target_1):
            stored_arr.append([p, t])
        # print("nums: ", nums)
        # break
    test_auc = calc_auc(stored_arr)
    accuracy_sum = accuracy_sum / nums
    loss_sum = loss_sum / nums
    aux_loss_sum / nums
    global best_auc
    if best_auc < test_auc:
        best_auc = test_auc
        if args.mode == 'train':
            model.save(sess, model_path)
    return test_auc, loss_sum, accuracy_sum, aux_loss_sum, eval_time, nums
def train_synthetic(   
        batch_size = 128,
        maxlen = 100,
        model_type = 'DNN',
        data_type = 'FP32',
        seed = 2,
        n_uid = 543060,
        n_mid = 100000 * 300,
        n_cat = 1601,
        embedding_device = 'gpu'      
):
    print("batch_size: ", batch_size)
    print("model: ", model_type)
    model_path = "dnn_save_path/ckpt_noshuff" + model_type + str(seed)
    best_model_path = "dnn_best_model/ckpt_noshuff" + model_type + str(seed)
    gpu_options = tf.GPUOptions(allow_growth=True)
    synthetic_input = True
    
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options,log_device_placement=False)) as sess:
        # parameters needs to put in config file
       
        if model_type == 'DNN':
            model = Model_DNN(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE, data_type = data_type, 
            synthetic_input = synthetic_input, batch_size = batch_size, max_length = maxlen, device = embedding_device)
        elif model_type == 'PNN':
            model = Model_PNN(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)
        elif model_type == 'Wide':
            model = Model_WideDeep(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)
        elif model_type == 'DIN':
            model = Model_DIN(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)
        elif model_type == 'DIN-V2-gru-att-gru':
            model = Model_DIN_V2_Gru_att_Gru(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)
        elif model_type == 'DIN-V2-gru-gru-att':
            model = Model_DIN_V2_Gru_Gru_att(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)
        elif model_type == 'DIN-V2-gru-qa-attGru':
            model = Model_DIN_V2_Gru_QA_attGru(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)
        elif model_type == 'DIN-V2-gru-vec-attGru':
            model = Model_DIN_V2_Gru_Vec_attGru(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)
        elif model_type == 'DIEN':
            model = Model_DIN_V2_Gru_Vec_attGru_Neg(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE, data_type, 
            synthetic_input = synthetic_input, batch_size = batch_size, max_length = maxlen, device = embedding_device)
        else:
            print ("Invalid model_type : %s", model_type)
            return
        
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        sys.stdout.flush()
        
        iter = 0
        train_size = 0
        approximate_accelerator_time = 0

        for itr in range(1):
            for i in range(500):   
                start_time = time.time()
                _, _, _ = model.train_synthetic_input(sess)
                end_time = time.time()
                # print("training time of one batch: %.3f" % (end_time - start_time))
                one_iter_time = end_time - start_time   
                approximate_accelerator_time += one_iter_time                 
                iter += 1
                sys.stdout.flush()
                if (iter % 100) == 0:
                    print('iter: %d ----> speed: %.4f  QPS' % 
                                        (iter, 1.0 * batch_size /one_iter_time ))    
         
        print("Total recommendations: %d" % (iter * batch_size))
        print("Approximate accelerator time in seconds is %.3f" % approximate_accelerator_time)
        print("Approximate accelerator performance in recommendations/second is %.3f" % (float(iter * batch_size)/float(approximate_accelerator_time)))

     
def train(
        train_file = "local_train_splitByUser",
        test_file = "local_test_splitByUser",
        uid_voc = "uid_voc.pkl",
        mid_voc = "mid_voc.pkl",
        cat_voc = "cat_voc.pkl",
        batch_size = 128,
        maxlen = 100,
        test_iter = 100,
        save_iter = 100,
        model_type = 'DNN',
        data_type = 'FP32',
	    seed = 2,
):
    print("batch_size: ", batch_size)
    print("model: ", model_type)
    model_path = "dnn_save_path/ckpt_noshuff" + model_type + str(seed)
    best_model_path = "dnn_best_model/ckpt_noshuff" + model_type + str(seed)
    gpu_options = tf.GPUOptions(allow_growth=True)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        train_data = DataIterator(train_file, uid_voc, mid_voc, cat_voc, batch_size, maxlen, shuffle_each_epoch=False)
        test_data = DataIterator(test_file, uid_voc, mid_voc, cat_voc, batch_size, maxlen)
        n_uid, n_mid, n_cat = train_data.get_n()
        print("Number of uid = %i, mid = %i, cat = %i" % (n_uid, n_mid, n_cat)) #Number of uid = 543060, mid = 367983, cat = 1601 for Amazon dataset
        if model_type == 'DNN':
            model = Model_DNN(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE, data_type = data_type, 
            batch_size = batch_size, max_length = maxlen)
        elif model_type == 'PNN':
            model = Model_PNN(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)
        elif model_type == 'Wide':
            model = Model_WideDeep(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)
        elif model_type == 'DIN':
            model = Model_DIN(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)
        elif model_type == 'DIN-V2-gru-att-gru':
            model = Model_DIN_V2_Gru_att_Gru(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)
        elif model_type == 'DIN-V2-gru-gru-att':
            model = Model_DIN_V2_Gru_Gru_att(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)
        elif model_type == 'DIN-V2-gru-qa-attGru':
            model = Model_DIN_V2_Gru_QA_attGru(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)
        elif model_type == 'DIN-V2-gru-vec-attGru':
            model = Model_DIN_V2_Gru_Vec_attGru(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)
        elif model_type == 'DIEN':
            model = Model_DIN_V2_Gru_Vec_attGru_Neg(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE, data_type, 
            batch_size = batch_size, max_length = maxlen)
        else:
            print ("Invalid model_type : %s", model_type)
            return
        # for var in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES):
        #     print("global variable dtype: ", var.dtype)
        #     if var.dtype == 'float32_ref':
        #         print("global variable: ", var)
        # model = Model_DNN(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        sys.stdout.flush()
        #print('test_auc: %.4f ---- test_loss: %.4f ---- test_accuracy: %.4f ---- test_aux_loss: %.4f ---- eval_time: %.3f ---- num_iters: %d' % eval(sess, test_data, model, best_model_path))
        sys.stdout.flush()

        iter = 0
        lr = 0.001
        train_size = 0
        approximate_accelerator_time = 0

        for itr in range(1):
            loss_sum = 0.0
            accuracy_sum = 0.
            aux_loss_sum = 0.
            for src, tgt in train_data:
                
                uids, mids, cats, mid_his, cat_his, mid_mask, target, sl, noclk_mids, noclk_cats = prepare_data(src, tgt, maxlen, return_neg=True)
                start_time = time.time()
                loss, acc, aux_loss = model.train(sess, [uids, mids, cats, mid_his, cat_his, mid_mask, target, sl, lr, noclk_mids, noclk_cats])
                end_time = time.time()
                # print("training time of one batch: %.3f" % (end_time - start_time))
                approximate_accelerator_time += end_time - start_time
                loss_sum += loss
                accuracy_sum += acc
                aux_loss_sum += aux_loss
                iter += 1
                train_size += batch_size
                sys.stdout.flush()
                if (iter % test_iter) == 0:
                    # print("train_size: %d" % train_size)
                    # print("approximate_accelerator_time: %.3f" % approximate_accelerator_time)
                    print('iter: %d ----> train_loss: %.4f ---- train_accuracy: %.4f ---- train_aux_loss: %.4f' % \
                                        (iter, loss_sum / test_iter, accuracy_sum / test_iter, aux_loss_sum / test_iter))
                    print(' test_auc: %.4f ----test_loss: %.4f ---- test_accuracy: %.4f ---- test_aux_loss: %.4f ---- eval_time: %.3f ---- num_iters: %d' % eval(sess, test_data, model, best_model_path))
                    loss_sum = 0.0
                    accuracy_sum = 0.0
                    aux_loss_sum = 0.0
                if (iter % save_iter) == 0:
                    print('save model iter: %d' %(iter))
                    model.save(sess, model_path+"--"+str(iter))
                if train_size >= TOTAL_TRAIN_SIZE:
                    break
            lr *= 0.5
            if train_size >= TOTAL_TRAIN_SIZE:
                break
        print("iter: %d" % iter)
        print("Total recommendations: %d" % TOTAL_TRAIN_SIZE)
        print("Approximate accelerator time in seconds is %.3f" % approximate_accelerator_time)
        print("Approximate accelerator performance in recommendations/second is %.3f" % (float(TOTAL_TRAIN_SIZE)/float(approximate_accelerator_time)))

def test(
        train_file = "local_train_splitByUser",
        test_file = "local_test_splitByUser",
        uid_voc = "uid_voc.pkl",
        mid_voc = "mid_voc.pkl",
        cat_voc = "cat_voc.pkl",
        batch_size = 128,
        maxlen = 100,
        model_type = 'DNN',
        data_type = 'FP32',
	    seed = 2
):
    print("batch_size: ", batch_size)
    print("model: ", model_type)
    model_path = "dnn_best_model_trained/ckpt_noshuff" + model_type + str(seed)
    gpu_options = tf.GPUOptions(allow_growth=True)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        train_data = DataIterator(train_file, uid_voc, mid_voc, cat_voc, batch_size, maxlen)
        test_data = DataIterator(test_file, uid_voc, mid_voc, cat_voc, batch_size, maxlen)
        n_uid, n_mid, n_cat = train_data.get_n()
        if model_type == 'DNN':
            model = Model_DNN(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)
        elif model_type == 'PNN':
            model = Model_PNN(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)
        elif model_type == 'Wide':
            model = Model_WideDeep(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)
        elif model_type == 'DIN':
            model = Model_DIN(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)
        elif model_type == 'DIN-V2-gru-att-gru':
            model = Model_DIN_V2_Gru_att_Gru(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)
        elif model_type == 'DIN-V2-gru-gru-att':
            model = Model_DIN_V2_Gru_Gru_att(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)
        elif model_type == 'DIN-V2-gru-qa-attGru':
            model = Model_DIN_V2_Gru_QA_attGru(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)
        elif model_type == 'DIN-V2-gru-vec-attGru':
            model = Model_DIN_V2_Gru_Vec_attGru(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)
        elif model_type == 'DIEN':
            model = Model_DIN_V2_Gru_Vec_attGru_Neg(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE, data_type)
        else:
            print ("Invalid model_type : %s", model_type)
            return
        # for var in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES):
        #     print("global variable: ", var)
        if data_type == 'FP32':
            model.restore(sess, model_path)
        if data_type == 'FP16':
            fp32_variables = [var_name for var_name, _ in tf.contrib.framework.list_variables(model_path)]
            #print("fp32_variables: ", fp32_variables)
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())
            for variable in tf.global_variables():
                #print("variable: ", variable)
                if variable.op.name in fp32_variables:
                    var = tf.contrib.framework.load_variable(model_path, variable.op.name)
                    # print("var: ", var)
                    # print("var.dtype: ", var.dtype)
                    if(variable.dtype == 'float16_ref'):
                        tf.add_to_collection('assignOps', variable.assign(tf.cast(var, tf.float16)))
                        # print("var value: ", sess.run(tf.cast(var, tf.float16)))
                    else:
                        tf.add_to_collection('assignOps', variable.assign(var))
                else:
                    raise ValueError("Variable %s is missing from checkpoint!" % variable.op.name)
            sess.run(tf.get_collection('assignOps'))
            # for variable in sess.run(tf.get_collection('assignOps')):
            #     print("after load checkpoint: ", variable)
        # for variable in tf.global_variables():
        #     print("after load checkpoint: ", sess.run(variable))
        approximate_accelerator_time = 0
        test_auc, test_loss, test_accuracy, test_aux_loss, eval_time, num_iters = eval(sess, test_data, model, model_path)
        approximate_accelerator_time += eval_time
        print('test_auc: %.4f ----test_loss: %.4f ---- test_accuracy: %.9f ---- test_aux_loss: %.4f ---- eval_time: %.3f' % (test_auc, test_loss, test_accuracy, test_aux_loss, eval_time))
        test_auc, test_loss, test_accuracy, test_aux_loss, eval_time, num_iters = eval(sess, test_data, model, model_path)
        approximate_accelerator_time += eval_time
        print('test_auc: %.4f ----test_loss: %.4f ---- test_accuracy: %.9f ---- test_aux_loss: %.4f ---- eval_time: %.3f' % (test_auc, test_loss, test_accuracy, test_aux_loss, eval_time))
        test_auc, test_loss, test_accuracy, test_aux_loss, eval_time, num_iters = eval(sess, test_data, model, model_path)
        approximate_accelerator_time += eval_time
        print('test_auc: %.4f ----test_loss: %.4f ---- test_accuracy: %.9f ---- test_aux_loss: %.4f ---- eval_time: %.3f' % (test_auc, test_loss, test_accuracy, test_aux_loss, eval_time))
        test_auc, test_loss, test_accuracy, test_aux_loss, eval_time, num_iters = eval(sess, test_data, model, model_path)
        approximate_accelerator_time += eval_time
        print('test_auc: %.4f ----test_loss: %.4f ---- test_accuracy: %.9f ---- test_aux_loss: %.4f ---- eval_time: %.3f' % (test_auc, test_loss, test_accuracy, test_aux_loss, eval_time))
        test_auc, test_loss, test_accuracy, test_aux_loss, eval_time, num_iters = eval(sess, test_data, model, model_path)
        approximate_accelerator_time += eval_time
        print('test_auc: %.4f ----test_loss: %.4f ---- test_accuracy: %.9f ---- test_aux_loss: %.4f ---- eval_time: %.3f' % (test_auc, test_loss, test_accuracy, test_aux_loss, eval_time))
        print("Total recommendations: %d" % (num_iters*batch_size))
        print("Approximate accelerator time in seconds is %.3f" % approximate_accelerator_time)
        print("Approximate accelerator performance in recommendations/second is %.3f" % (float(5*num_iters*batch_size)/float(approximate_accelerator_time)))

if __name__ == '__main__':
    SEED = args.seed
    tf.set_random_seed(SEED)
    numpy.random.seed(SEED)
    random.seed(SEED)
    if args.mode == 'train':
        train(model_type=args.model, seed=SEED, batch_size=args.batch_size, data_type=args.data_type)
    elif args.mode == 'test':
        test(model_type=args.model, seed=SEED, batch_size=args.batch_size, data_type=args.data_type)
    elif args.mode == 'synthetic':
        train_synthetic(model_type=args.model, seed=SEED, batch_size=args.batch_size, 
        data_type=args.data_type, embedding_device = args.embedding_device
        ) 
    else:
        print('do nothing...')


