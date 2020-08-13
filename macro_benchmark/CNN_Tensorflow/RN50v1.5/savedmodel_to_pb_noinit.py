import tensorflow as tf
from tensorflow.python.saved_model import loader_impl

import sys
import pdb

def load_model(model_path):
    saved_model = loader_impl._parse_saved_model(model_path)
    tf.reset_default_graph()
    cfg = tf.ConfigProto(allow_soft_placement=True,
                         log_device_placement=False)
    cfg.gpu_options.allow_growth = True
    with tf.Session(config=cfg) as sess:
        tags = saved_model.meta_graphs[0].meta_info_def.tags
        meta_graph_def = tf.saved_model.loader.load(sess, tags, model_path)
        sdef_key = [key for key in meta_graph_def.signature_def.keys()][0]
        tmp_outputs = meta_graph_def.signature_def[sdef_key].outputs.values()
        model_outputs = [v.name[:-2] for v in tmp_outputs]
        graph_def = tf.graph_util.convert_variables_to_constants(
            sess, sess.graph_def, model_outputs)
    graph_def = tf.graph_util.extract_sub_graph(graph_def, model_outputs)
    for i,node in enumerate(graph_def.node):
        if '_class' in node.attr.keys():
            node.attr.pop('_class')
    return graph_def

if __name__ == '__main__':
    graph_def = load_model(sys.argv[1])
    with tf.gfile.FastGFile('frozen.pb', mode='wb') as f:
        f.write(graph_def.SerializeToString())
