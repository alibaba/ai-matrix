import tensorflow as tf
import time
import argparse
import shlex
from functools import reduce

class ConvInstance:
    def __init__(self, input_dim, filter_dim, stride):
        self.input_dim = input_dim
        self.filter_dim = filter_dim
        self.stride = [1, stride, stride, 1]

    def get_input_dim(self):
        return self.input_dim

    def get_filter_dim(self):
        return self.filter_dim

    def get_stride(self):
        return self.stride
    
class FcInstance:
    def __init__(self, input_dim, num_outputs):
        self.input_dim = input_dim
        self.num_outputs = num_outputs

    def get_input_dim(self):
        return self.input_dim

    def get_num_outputs(self):
        return self.num_outputs

class PoolInstance:
    def __init__(self, input_dim, kernel_size, S_H, S_W):
        self.input_dim = input_dim 
        self.kernel_size = [1, kernel_size, kernel_size, 1]
        self.stride = [1, S_H, S_W, 1]

    def get_input_dim(self):
        return self.input_dim

    def get_kernel_size(self):
        return self.kernel_size

    def get_stride(self):
        return self.stride

class MaxPoolInstance(PoolInstance):
    def __init__(self, input_dim, kernel_size, S_H, S_W):
        PoolInstance.__init__(self, input_dim, kernel_size, S_H, S_W)

class AvgPoolInstance(PoolInstance):
    def __init__(self, input_dim, kernel_size, S_H, S_W):
        PoolInstance.__init__(self, input_dim, kernel_size, S_H, S_W)

class BnInstance:
    def __init__(self, input_dim, mean, variance, offset, scale, variance_epsilon):
        self.input_dim = input_dim
        self.mean = mean
        self.variance = variance
        self.offset = offset
        self.scale = scale
        self.variance_epsilon = variance_epsilon

    def get_mean(self):
        return self.mean

    def get_variance(self):
        return self.variance

    def get_offset(self):
        return self.offset

    def get_scale(self):
        return self.scale

    def get_variance_epsilon(self):
        return self.variance_epsilon

    def set_input_dim(self, input_dim):
        self.input_dim = input_dim

    def get_input_dim(self):
        return self.input_dim


class LrnInstance:
    def __init__(self, input_dim, depth_radius, bias, alpha, beta):
        self.input_dim = input_dim
        self.depth_radius = depth_radius
        self.bias = bias
        self.alpha = alpha
        self.beta = beta

    def get_depth_radius(self):
        return self.depth_radius

    def get_bias(self):
        return self.bias

    def get_alpha(self):
        return self.alpha

    def get_beta(self):
        return self.beta

    def get_input_dim(self):
        return self.input_dim

    def set_input_dim(self, input_dim):
        self.input_dim = input_dim

class ActiveInstance:
    def __init__(self, feature_dim):
        self.feature_dim = feature_dim

    def get_feature_dim(self):
        return self.feature_dim

class ReluInstance(ActiveInstance):
    def __init__(self, feature_dim):
        ActiveInstance.__init__(self, feature_dim)

class SigmoidInstance(ActiveInstance):
    def __init__(self, feature_dim):
        ActiveInstance.__init__(self, feature_dim)

class SoftmaxInstance(ActiveInstance):
    def __init__(self, feature_dim):
        ActiveInstance.__init__(self, feature_dim)

class TanhInstance(ActiveInstance):
    def __init__(self, feature_dim):
        ActiveInstance.__init__(self, feature_dim)

class TestCase():
    def __init__(self):
        self.test_case_list = []

    def convert_list(self, src_list):
        return list(map(int, src_list[1:-1].split(',')))

class ConvTestCase(TestCase):
    def __init__(self):
        TestCase.__init__(self)
        self.index = {'input_dim': 0, 'filter_dim': 1, 'stride': 2}

    def gen_test_case(self, param_file):
        with open(param_file) as lines:
            for line in lines:
                params = shlex.split(line)
                input_dim = params[self.index['input_dim']]
                filter_dim = params[self.index['filter_dim']]
                stride = int(params[self.index['stride']])
                input_dim = self.convert_list(input_dim)
                filter_dim = self.convert_list(filter_dim)
                test_case = ConvInstance(input_dim, filter_dim, stride)
                self.test_case_list.append(test_case)

        return self.test_case_list

class FcTestCase(TestCase):
    def __init__(self):
        TestCase.__init__(self)
        self.index = {'input_dim': 0, 'num_outputs': 1}

    def gen_test_case(self, param_file):
        input_dim_list = []
        with open(param_file) as lines:
            for line in lines:
                params = shlex.split(line)
                input_dim = params[self.index['input_dim']]
                input_dim = self.convert_list(input_dim)
                num_outputs = int(params[self.index['num_outputs']])
                input_dim.append(num_outputs)
                input_dim_list.append(input_dim)
         
        #sort input dimensions
        input_dim_list.sort(key=lambda dim: reduce(lambda x,y:x*y,dim,1))
        for input_dim in input_dim_list:
                test_case = FcInstance(input_dim[:-1], input_dim[-1])
                self.test_case_list.append(test_case)

        return self.test_case_list

class PoolTestCase(TestCase):
    def __init__(self):
        TestCase.__init__(self)
        self.index = {'input_dim': 0, 'kernel_size': 1, 'stride_h': 2, 'stride_w': 3}

    def gen_test_case(self, param_file):
        with open(param_file) as lines:
            for line in lines:
                params = shlex.split(line)
                input_dim = params[self.index['input_dim']]
                input_dim = self.convert_list(input_dim)
                kernel_size = int(params[self.index['kernel_size']])
                stride_h = int(params[self.index['stride_h']])
                stride_w = int(params[self.index['stride_w']])
                test_case = MaxPoolInstance(input_dim, kernel_size, stride_h, stride_w)
                self.test_case_list.append(test_case)

        return self.test_case_list

class BnTestCase(TestCase):
    def __init__(self):
        TestCase.__init__(self)
        self.index = {'input_dim': 0}

    def gen_test_case(self, param_file):
        input_dim_list= []
        with open(param_file) as lines:
            for line in lines:
                params = shlex.split(line)
                input_dim = params[self.index['input_dim']]
                input_dim = self.convert_list(input_dim)
                input_dim_list.append(input_dim)

        #sort input dimensions
        input_dim_list.sort(key=lambda dim: reduce(lambda x,y:x*y,dim,1))
        for input_dim in input_dim_list:
                test_case = BnInstance(input_dim, 0.35, 0.55, 2.3, 8.6, 0.03)
                self.test_case_list.append(test_case)

        return self.test_case_list

class LrnTestCase(TestCase):
    def __init__(self):
        TestCase.__init__(self)
        self.index = {'input_dim': 0}

    def gen_test_case(self, param_file):
        input_dim_list = []
        with open(param_file) as lines:
            for line in lines:
                params = shlex.split(line)
                input_dim = params[self.index['input_dim']]
                input_dim = self.convert_list(input_dim)
                input_dim_list.append(input_dim)

        #sort input dimensions
        input_dim_list.sort(key=lambda dim: reduce(lambda x,y:x*y,dim,1))
        for input_dim in input_dim_list:
            for depth_radius in range(1, 8):
                test_case = LrnInstance(input_dim, depth_radius, 1, 1e-4, 0.75)
                self.test_case_list.append(test_case)

        return self.test_case_list

class ActiveTestCase(TestCase):
    def __init__(self):
        TestCase.__init__(self)
        self.index = {'input_dim': 0}

    def gen_test_case(self, param_file):
        input_dim_list = []
        with open(param_file) as lines:
            for line in lines:
                params = shlex.split(line)
                input_dim = params[self.index['input_dim']]
                input_dim = self.convert_list(input_dim)
                input_dim_list.append(input_dim) 

        #sort input dimensions
        input_dim_list.sort(key=lambda dim: reduce(lambda x,y:x*y,dim,1))
        for input_dim in input_dim_list:
            test_case = ActiveInstance(input_dim)
            self.test_case_list.append(test_case)

        return self.test_case_list

class CnnOps:
    def __init__(self, times):
        self.times = times

    def conv(self, inst, log):
        print('conv op')
        #get input and filter dimensions and stride of current test instance
        input_dim = inst.get_input_dim()
        filter_dim = inst.get_filter_dim()
        stride = inst.get_stride()

        #create input data and weights
        op_input = tf.Variable(tf.random_normal(input_dim))
        op_weights = tf.Variable(tf.random_normal(filter_dim))

        #define op
        op = tf.nn.conv2d(op_input, op_weights, stride, data_format='NHWC', padding='VALID')

        #warm up op
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        sess.run(op)

        #repeat op
        start = time.time()
        for i in range(0, self.times):
            sess.run(op)
        end = time.time()
        duration = end - start
        avg = 1000*duration/times
        #print("avg time(s): %.3f\n" %(duration/times))

        #log record
        log.write('%-25s, %-25s, %-15s, %-25.3f\n' %(input_dim, filter_dim, stride, avg))
        log.flush()
        sess.close()

    def fc(self, inst, log):
        print('fc op')
        #get input dimensions and num_outputs of inst
        input_dim = inst.get_input_dim()
        num_outputs = inst.get_num_outputs()
        N = input_dim[0]
        D = int(reduce(lambda x,y:x*y, input_dim) / N)
        new_input_dim = [N, D]
        weight_dim = [D, num_outputs]

        #create input data and weights for inst
        op_data = tf.Variable(tf.random_normal(input_dim))
        op_input = tf.reshape(op_data, new_input_dim)
        op_weights = tf.Variable(tf.random_uniform(weight_dim,minval=-1,maxval=1,dtype=tf.float32))

        #define op
        op = tf.matmul(op_input, op_weights)

        #warm up op
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        sess.run(op)

        #repeat op 
        start = time.time()
        for i in range(0, self.times):
            sess.run(op)
        end = time.time()
        duration = end -start
        avg = 1000*duration/times

        #log record
        log.write('%-25s, %-15s, %-25.3f\n' %(input_dim, num_outputs, avg))
        log.flush()
        sess.close()

    def max_pool(self, inst, log):
        print('max_pool op')
        #get input dimensions, kernel size and stride
        input_dim = inst.get_input_dim()
        kernel_size = inst.get_kernel_size() 
        stride = inst.get_stride() 

        #create input data
        op_data = tf.Variable(tf.random_normal(input_dim))

        #define op
        op = tf.nn.max_pool(op_data, kernel_size, stride, 'VALID', 'NHWC')

        #warm up op
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        sess.run(op)
        
        #repeat op
        start = time.time()
        for i in range(0, self.times):
            sess.run(op)
        end = time.time()
        duration = end - start
        avg = 1000*duration/times

        #log record
        log.write('%-25s, %-25s, %-15s, %-25.3f\n' %(input_dim, kernel_size, stride, avg))
        log.flush()
        sess.close()

    def avg_pool(self, inst, log):
        print('avg_pool op')
        #get input dimensions, kernel size and stride
        input_dim = inst.get_input_dim()
        kernel_size = inst.get_kernel_size()
        stride = inst.get_stride()

        #create input data
        op_data = tf.Variable(tf.random_normal(input_dim))
        
        #define op
        op = tf.nn.avg_pool(op_data, kernel_size, stride, 'VALID', 'NHWC')

        #warm up op
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        sess.run(op)

        #repeat op
        start = time.time()
        for i in range(0, self.times):
            sess.run(op)
        end = time.time()
        duration = end - start
        avg = 1000*duration/times

        #log record
        log.write('%-25s, %-25s, %-15s, %-25.3f\n' %(input_dim, kernel_size, stride, avg))
        log.flush()
        sess.close()

    def bn(self, inst, log):
        print('bn op')
        #get input dimensions and parameters
        input_dim = inst.get_input_dim()
        mean = inst.get_mean() 
        scale = inst.get_scale() 
        variance = inst.get_variance()
        offset = inst.get_offset() 
        variance_epsilon = inst.get_variance_epsilon()

        #create input data
        x = tf.Variable(tf.random_uniform(input_dim, minval= -1, maxval=1, dtype=tf.float32))

        #define op
        op = tf.nn.batch_normalization(x, mean, variance, offset, scale, variance_epsilon)

        #warm up op
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        sess.run(op)
        
        #repeat op
        start = time.time()
        for i in range(0, times):
            sess.run(op)
        end = time.time()
        duration = end - start
        avg = 1000*duration/times

        #log record
        log.write('%-25s, %-25.3f\n' %(input_dim, avg))
        log.flush()
        sess.close()

    def lrn(self, inst, log):
        print('lrn op')
        #get input dimensions, depth radius, and parameters
        input_dim = inst.get_input_dim()
        depth_radius = inst.get_depth_radius() 
        alpha = inst.get_alpha() 
        bias = inst.get_bias() 
        beta = inst.get_beta()

        #create input data
        op_data = tf.Variable(tf.random_normal(input_dim))

        #define op
        op = tf.nn.local_response_normalization(op_data, depth_radius, bias, alpha, beta)

        #warm up op
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        sess.run(op)

        #repeat op
        start = time.time()
        for i in range(0, self.times):
            sess.run(op)
        end = time.time()
        duration = end - start
        avg = 1000*duration/times

        #log record
        log.write('%-25s, %-15s, %-25.3f\n' %(input_dim, depth_radius, avg))
        log.flush()
        sess.close()

    def relu(self, inst, log):
        print('relu op')
        #get input dimensions
        feature_dim = inst.get_feature_dim()

        #create input data
        features = tf.Variable(tf.random_uniform(feature_dim, minval=-1, maxval=1, dtype=tf.float32))
        #print(sess.run(features))

        #define op
        op = tf.nn.relu(features)

        #warm up op
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        res = sess.run(op)
        
        #repeat op
        start = time.time() 
        for i in range(0, self.times):
            res = sess.run(op)
            #print(res)
        end = time.time()
        duration = end - start
        avg = 1000*duration/times

        #log record
        log.write('%-25s, %-25.3f\n' %(feature_dim, avg))
        log.flush()
        sess.close()

    def sigmoid(self, inst, log):
        print('sigmoid op')
        #get input dimensions
        feature_dim = inst.get_feature_dim() 

        #create input data
        x = tf.Variable(tf.random_normal(feature_dim))
        #print(sess.run(x))
        
        #define op
        op = tf.nn.sigmoid(x)
        
        #warm up op
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        res = sess.run(op)
        
        #repeat op
        start = time.time()
        for i in range(0, self.times):
            res = sess.run(op)
            #print(res)
        end = time.time()
        duration = end - start
        avg = 1000*duration/times

        #log record
        log.write('%-25s, %-25.3f\n' %(feature_dim, avg))
        log.flush()
        sess.close()
        
    def softmax(self, inst, log):
        print('softmax op')
        #get input dimensions
        feature_dim = inst.get_feature_dim()
        
        #create input data
        x = tf.Variable(tf.random_normal(feature_dim))
        #print(sess.run(x))
        
        #define op
        op = tf.nn.softmax(x)

        #warm up op
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        res = sess.run(op)
        
        #repeat op
        start = time.time()
        for i in range(0, self.times):
            res = sess.run(op)
            #print(res)
        end = time.time()
        duration = end - start
        avg = 1000*duration/times

        #log record
        log.write('%-25s, %-25.3f\n' %(feature_dim, avg))
        log.flush()
        sess.close()

    def tanh(self, inst, log):
        print('tanh')
        #get input dimensions
        feature_dim = inst.get_feature_dim()

        #create input data
        x = tf.Variable(tf.random_normal(feature_dim))
        #print(sess.run(x))
        
        #define op
        op = tf.tanh(x)

        #warm up op
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        res = sess.run(op)
        
        #repeat op
        start = time.time()
        for i in range(0, self.times):
            res = sess.run(op)
            #print(res)
        end = time.time()
        duration = end - start
        avg = 1000*duration/times

        #log record
        log.write('%-25s, %-25.3f\n' %(feature_dim, avg))
        log.flush()
        sess.close()

class CnnOpTest():
    def __init__(self, op_name, times, interval):
        self.op_name = op_name
        self.op = CnnOps(times) 
        self.times = times
        self.interval = interval
        self.log = open('log/'+op_name+'.csv', 'w')

    def run(self):
        if self.op_name == 'conv':
            self.test_conv()
        elif self.op_name == 'fc':
            self.test_fc()
        elif self.op_name == 'max_pool':
            self.test_max_pool()
        elif self.op_name == 'avg_pool':
            self.test_avg_pool()
        elif self.op_name == 'bn':
            self.test_bn()
        elif self.op_name == 'lrn':
            self.test_lrn()
        elif self.op_name == 'relu':
            self.test_relu()
        elif self.op_name == 'sigmoid':
            self.test_sigmoid()
        elif self.op_name == 'softmax':
            self.test_softmax()
        elif self.op_name == 'tanh':
            self.test_tanh()

    def test_conv(self):
        generator = ConvTestCase()
        test_case_list = generator.gen_test_case('params/conv.params')
        
        self.log.write('%-25s %-25s %-15s %-25s\n' %('input_dim', 'filter_dim', 'stride', 'avg time(ms)'))
        self.log.flush()
        for test_case in test_case_list:
            self.op.conv(test_case, self.log)
            time.sleep(self.interval)

    def test_fc(self):
        generator = FcTestCase()
        test_case_list = generator.gen_test_case('params/fc.params')

        self.log.write('%-25s %-15s %-25s\n' %('input_dim', 'num_outputs', 'avg time(ms)'))
        self.log.flush()
        for test_case in test_case_list:
            self.op.fc(test_case, self.log)
            time.sleep(self.interval)

    def test_max_pool(self):
        generator = PoolTestCase()
        test_case_list = generator.gen_test_case('params/pool.params')

        self.log.write('%-25s %-25s %-15s %-25s\n' %('input_dim', 'kernel_size', 'stride', 'avg time(ms)'))
        self.log.flush()
        for test_case in test_case_list:
            self.op.max_pool(test_case, self.log)
            time.sleep(self.interval)

    
    def test_avg_pool(self):
        generator = PoolTestCase()
        test_case_list = generator.gen_test_case('params/pool.params')

        self.log.write('%-25s %-25s %-15s %-25s\n' %('input_dim', 'kernel_size', 'stride', 'avg time(ms)'))
        self.log.flush()
        for test_case in test_case_list:
            self.op.avg_pool(test_case, self.log)
            time.sleep(self.interval)

    def test_bn(self):
        #inst = BnInstance([4096], 0.35, 0.55, 2.3, 8.6, 0.03)
        #self.op.bn(inst)
        generator = BnTestCase()
        test_case_list = generator.gen_test_case('params/normal.params')

        self.log.write('%-25s %-25s\n' %('input_dim', 'avg time(ms)'))
        self.log.flush()
        for test_case in test_case_list:
            self.op.bn(test_case, self.log)
            time.sleep(self.interval)

    def test_lrn(self):
        #inst = LrnInstance([64,1024,1024,3], 5, 1, 1e-4, 0.75)
        #self.op.lrn(inst)
        generator = LrnTestCase()
        test_case_list = generator.gen_test_case('params/normal.params')
        
        self.log.write('%-25s %-15s %-25s\n' %('input_dim', 'depth_radius', 'avg time(ms)'))
        self.log.flush()
        for test_case in test_case_list:
            self.op.lrn(test_case, self.log)
            time.sleep(self.interval)

    def test_relu(self):
        generator = ActiveTestCase()
        test_case_list = generator.gen_test_case('params/active.params')

        self.log.write('%-25s %-25s\n' %('input_dim', 'avg time(ms)'))
        self.log.flush()
        for test_case in test_case_list:
            self.op.relu(test_case, self.log)
            time.sleep(self.interval)

    def test_sigmoid(self):
        generator = ActiveTestCase()
        test_case_list = generator.gen_test_case('params/active.params')

        self.log.write('%-25s %-25s\n' %('input_dim', 'avg time(ms)'))
        self.log.flush()
        for test_case in test_case_list:
            self.op.sigmoid(test_case, self.log)
            time.sleep(self.interval)

    def test_softmax(self):
        generator = ActiveTestCase()
        test_case_list = generator.gen_test_case('params/active.params')

        self.log.write('%-25s %-25s\n' %('input_dim', 'avg time(ms)'))
        self.log.flush()
        for test_case in test_case_list:
            self.op.softmax(test_case, self.log)
            time.sleep(self.interval)

    def test_tanh(self):
        generator = ActiveTestCase()
        test_case_list = generator.gen_test_case('params/active.params')

        self.log.write('%-25s %-25s\n' %('input_dim', 'avg time(ms)'))
        self.log.flush()
        for test_case in test_case_list:
            self.op.tanh(test_case, self.log)
            time.sleep(self.interval)

def main(op_name, times, interval):
    cnn_test = CnnOpTest(op_name, times, interval)
    cnn_test.run()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument('-op_name', default='conv', type=str, help='op name')
    parser.add_argument('-times', default='1', type=int, help='repeat times')
    parser.add_argument('-interval', default='1', type=int, help='time interval')
    args = parser.parse_args()
    op_name = args.op_name
    times = args.times
    interval = args.interval
    main(op_name, times, interval)
