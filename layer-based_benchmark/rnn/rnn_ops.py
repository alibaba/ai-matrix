import tensorflow as tf
import time
import argparse

class RnnInstance:
    def __init__(self, input_dim, num_units):
        self.input_dim = input_dim
        self.num_units = num_units

    def get_input_dim(self):
        return self.input_dim

    def get_num_units(self):
        return self.num_units

class RnnTestCase:
    def __init__(self):
        self.test_case_list = []
        self.vector_len = 64
                
    def gen_test_case(self, min_input_dim, max_input_dim, stride_input, min_units, max_units, stride_units):
        min_bs = min_input_dim[0]
        max_bs = max_input_dim[0]
        min_sl = min_input_dim[1]
        max_sl = max_input_dim[1]
        stride_bs = stride_input[0]
        stride_sl = stride_input[1]
        vl = self.vector_len

        for bs in range(min_bs, max_bs + 1, stride_bs):
            for sl in range(min_sl, max_sl + 1, stride_sl): 
                input_dim = [bs, sl, vl]
                for num_units in range(min_units, max_units + 1, stride_units): 
                    test_case = RnnInstance(input_dim, num_units)
                    self.test_case_list.append(test_case)

        return self.test_case_list

class RnnOps:
    def __init__(self, times):
        self.times = times
        self.time_diff = 0.1 #ms

    def lstm(self, inst, log):
        #get input dimensions and units num
        input_dim = inst.get_input_dim()
        num_units = inst.get_num_units()
        #create input data
        inputs = tf.Variable(tf.random_normal(input_dim))
        #define cell
        cell = tf.contrib.rnn.BasicLSTMCell(num_units)
        outputs, last_states = tf.nn.dynamic_rnn(cell=cell, dtype=tf.float32, inputs=inputs)

        #warm up cell
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        sess.run({'outputs': outputs, 'last_states': last_states}, feed_dict=None)
        tf.reset_default_graph()

        #repeat cell until time convergence with respect to time diff
        start = time.time()
        sess.run({'outputs': outputs, 'last_states': last_states}, feed_dict=None)
        end = time.time()
        duration = end - start
        pre_avg = 1000*duration

        while_times = 1
        while True:
            start = time.time()
            sess.run({'outputs': outputs, 'last_states': last_states}, feed_dict=None)
            end = time.time()
            duration = end - start
            avg = 1000*duration

            if abs(avg - pre_avg) <= self.time_diff:
                break
            if while_times >= self.times:
                break
            print('pre_avg %.3f, avg %.3f\n' %(pre_avg, avg))
            pre_avg = avg
            while_times = while_times + 1

        total_times = while_times
        log.write('%-15s %-15s %-15s %-15s %-15.3f %-15s\n' %(input_dim[0], input_dim[1], input_dim[2], num_units, avg, total_times))
        log.flush()
        sess.close()

    def gru(self, inst, log):
        #get input dimensions and units num
        input_dim = inst.get_input_dim()
        num_units = inst.get_num_units()
        #create input data
        inputs = tf.Variable(tf.random_normal(input_dim))
        #define cell
        cell = tf.contrib.rnn.GRUCell(num_units)
        outputs, last_states = tf.nn.dynamic_rnn(cell=cell, dtype=tf.float32, inputs=inputs)

        #warm up cell
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        sess.run({'outputs': outputs, 'last_states': last_states}, feed_dict=None)
        tf.reset_default_graph()

        #repeat cell until time convergence with respect to time diff
        start = time.time()
        sess.run({'outputs': outputs, 'last_states': last_states}, feed_dict=None)
        end = time.time()
        duration = end - start
        pre_avg = 1000*duration

        while_times = 1
        while True:
            start = time.time()
            sess.run({'outputs': outputs, 'last_states': last_states}, feed_dict=None)
            end = time.time()
            duration = end - start
            avg = 1000*duration

            if abs(avg - pre_avg) <= self.time_diff:
                break
            if while_times >= self.times:
                break
            print('pre_avg %.3f, avg %.3f\n' %(pre_avg, avg))
            pre_avg = avg
            while_times = while_times + 1

        total_times = while_times
        log.write('%-15s %-15s %-15s %-15s %-15.3f %-15s\n' %(input_dim[0], input_dim[1], input_dim[2], num_units, avg, total_times))
        log.flush()
        sess.close()

    def rnn(self, inst, log):
        #get input dimensions and units num
        input_dim = inst.get_input_dim()
        num_units = inst.get_num_units()
        #create input data
        inputs = tf.Variable(tf.random_normal(input_dim))
        #define cell
        cell = tf.contrib.rnn.BasicRNNCell(num_units)
        outputs, last_states = tf.nn.dynamic_rnn(cell=cell, dtype=tf.float32, inputs=inputs)

        #warm up cell
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        sess.run({'outputs': outputs, 'last_states': last_states}, feed_dict=None)
        tf.reset_default_graph()

        #repeat cell until time convergence with respect to time diff
        start = time.time()
        sess.run({'outputs': outputs, 'last_states': last_states}, feed_dict=None)
        end = time.time()
        duration = end - start
        pre_avg = 1000*duration

        while_times = 1
        while True:
            start = time.time()
            sess.run({'outputs': outputs, 'last_states': last_states}, feed_dict=None)
            end = time.time()
            duration = end - start
            avg = 1000*duration

            if abs(avg - pre_avg) <= self.time_diff:
                break
            if while_times >= self.times:
                break
            print('pre_avg %.3f, avg %.3f\n' %(pre_avg, avg))
            pre_avg = avg
            while_times = while_times + 1

        total_times = while_times
        log.write('%-15s %-15s %-15s %-15s %-15.3f %-15s\n' %(input_dim[0], input_dim[1], input_dim[2], num_units, avg, total_times))
        log.flush()
        sess.close()

class RnnOpTest:
    def __init__(self, op_name, times, interval, min_bs, max_bs, stride_bs, min_sl, max_sl, stride_sl, min_units, max_units, stride_units):
        self.op_name = op_name
        self.op = RnnOps(times)
        self.times = times
        self.interval = interval
        self.min_input_dim = [min_bs, min_sl]
        self.max_input_dim = [max_bs, max_sl]
        self.stride_input = [stride_bs, stride_sl]
        self.min_units = min_units
        self.max_units = max_units
        self.stride_units = stride_units
        self.log = open('log/'+op_name+'.log', 'w')

    def run(self):
        if self.op_name == 'lstm':
            self.test_lstm()
        elif self.op_name == 'gru':
            self.test_gru()
        elif self.op_name == 'rnn':
            self.test_rnn()

    def test_lstm(self):
        generator = RnnTestCase()
        test_case_list = generator.gen_test_case(self.min_input_dim, self.max_input_dim, self.stride_input, self.min_units, self.max_units, self.stride_units)
        self.print_title()
        #iterate each test case
        for test_case in test_case_list:
            self.op.lstm(test_case, self.log)
            time.sleep(self.interval)

    def test_gru(self):
        generator = RnnTestCase()
        test_case_list = generator.gen_test_case(self.min_input_dim, self.max_input_dim, self.stride_input, self.min_units, self.max_units, self.stride_units)
        self.print_title()
        #iterate each test case
        for test_case in test_case_list:
            self.op.gru(test_case, self.log)
            time.sleep(self.interval)

    def test_rnn(self):
        generator = RnnTestCase()
        test_case_list = generator.gen_test_case(self.min_input_dim, self.max_input_dim, self.stride_input, self.min_units, self.max_units, self.stride_units)
        self.print_title()
        #iterate each test case
        for test_case in test_case_list:
            self.op.rnn(test_case, self.log)
            time.sleep(self.interval)

    def print_title(self):
        self.log.write('%-15s %-15s %-15s %-15s %-15s %-15s\n' %('batch_size', 'seq_len', 'vector_len', 'hidden_units', 'avg time(ms)', 'total times'))

def main(args):
    op_name = args.op_name
    times = args.times
    interval = args.interval
    min_bs = args.min_bs
    max_bs = args.max_bs
    stride_bs = args.stride_bs
    min_sl = args.min_sl
    max_sl = args.max_sl
    stride_sl = args.stride_sl
    min_units = args.min_units
    max_units = args.max_units
    stride_units = args.stride_units

    rnn_test = RnnOpTest(op_name, times, interval, min_bs, max_bs, stride_bs, min_sl, max_sl, stride_sl, min_units, max_units, stride_units)
    rnn_test.run()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument('-op_name', default='lstm', type=str, help='op name')
    parser.add_argument('-times', default='1', type=int, help='repeat times')
    parser.add_argument('-interval', default='1', type=int, help='time interval')
    parser.add_argument('-min_bs', default='1', type=int, help='minimum batch size')
    parser.add_argument('-max_bs', default='1', type=int, help='max batch size')
    parser.add_argument('-stride_bs', default='1', type=int, help='batch size stride')
    parser.add_argument('-min_sl', default='1', type=int, help='minimum sequence length')
    parser.add_argument('-max_sl', default='1', type=int, help='max sequence length')
    parser.add_argument('-stride_sl', default='1', type=int, help='sequence length stride')
    parser.add_argument('-min_units', default='1', type=int, help='minimum hidden units')
    parser.add_argument('-max_units', default='1', type=int, help='max hidden units')
    parser.add_argument('-stride_units', default='1', type=int, help='hidden units stride')
    args = parser.parse_args()
    main(args)
