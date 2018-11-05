import caffe
import argparse

class LayerAnalyze:
    def __init__(self):
        self.param_bytes = 4
        self.name_list = []
        self.blob_list = []
        self.weight_list = []
        self.computation_list = []
        self.param_count_list = []

    def collect_name(self, name):
        self.name_list.append(name)

    def collect_blob(self, blob):
        self.blob_list.append(blob)

    def collect_weight(self, weight):
        self.weight_list.append(weight)

    def print_res(self):
        self.log.write('%-10s%-20s%-20s%-20s%-20s\n' %('layer', 'output dim', 'weight dim', 'computation(ops)', 'param count(bytes)'))
        self.log.flush()
        for (name, blob, weight, computation, param_count) in zip(self.name_list, self.blob_list, self.weight_list, self.computation_list, self.param_count_list):
            self.log.write('%-10s%-20s%-20s%-20d%-20d\n' %(name, str(blob), str(weight), computation, param_count))
            self.log.flush()

class ConvLayerAnalyze(LayerAnalyze):
    def __init__(self):
        LayerAnalyze.__init__(self)
        self.log = open('conv.log', 'w')

    def parse_layer(self):
        for (blob, weight) in zip(self.blob_list, self.weight_list):
            batch_size = blob[0]
            height = blob[2]
            width = blob[3]

            out_channel = weight[0]
            in_channel = weight[1]
            kernel_height = weight[2]
            kernel_width = weight[3]

            computation = batch_size * in_channel * height * width * kernel_height * kernel_width
            param_count = reduce(lambda x,y:x*y,weight,1) * self.param_bytes
            self.computation_list.append(computation)
            self.param_count_list.append(param_count)

class FCLayerAnalyze(LayerAnalyze):
    def __init__(self):
        LayerAnalyze.__init__(self)
        self.log = open('fc.log', 'w')

    def parse_layer(self):
        for (blob, weight) in zip(self.blob_list, self.weight_list):
            num_outputs = weight[1]
            computation = reduce(lambda x,y:x*y,blob,1) * num_outputs
            param_count = weight[0] * weight[1]
            self.computation_list.append(computation)
            self.param_count_list.append(param_count)

class NetAnalyze:
    def __init__(self, proto):
        self.net = caffe.Net(proto, caffe.TEST)
        self.conv_layer_analyze = ConvLayerAnalyze()
        self.fc_layer_analyze = FCLayerAnalyze()

    def traverse_blob(self):
        for layer_name, blob in self.net.blobs.iteritems():
            blob_shape = blob.data.shape
            if (layer_name.find('conv') != -1):
                self.conv_layer_analyze.collect_name(layer_name)
                self.conv_layer_analyze.collect_blob(blob_shape)
            elif (layer_name.find('fc') != -1):
                self.fc_layer_analyze.collect_name(layer_name)
                self.fc_layer_analyze.collect_blob(blob_shape)
    
    def traverse_param(self):
        for layer_name, params in self.net.params.iteritems():
            weight_shape = params[0].data.shape
            if (layer_name.find('conv') != -1):
                self.conv_layer_analyze.collect_weight(weight_shape)
            elif (layer_name.find('fc') != -1):
                self.fc_layer_analyze.collect_weight(weight_shape)

    def run_analyze(self):
        self.conv_layer_analyze.parse_layer()
        self.fc_layer_analyze.parse_layer()

    def print_res(self):
        self.conv_layer_analyze.print_res()
        self.fc_layer_analyze.print_res()

if __name__ == '__main__':
    caffe.set_mode_cpu()
    print('hello caffe')
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument('-proto', default='../proto/alexnet_deploy.prototxt', type=str, help='network prototxt')
    args = parser.parse_args()
    proto = args.proto
    net_analyze = NetAnalyze(proto) 
    net_analyze.traverse_blob()
    net_analyze.traverse_param()
    net_analyze.run_analyze()
    net_analyze.print_res()
