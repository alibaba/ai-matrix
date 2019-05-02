import tensorflow as tf

from Dice import dice

class Model(object):

  def __init__(self, user_count, item_count, cate_count, cate_list, predict_batch_size, predict_ads_num, num_accelerators, devices):

    tower_losses   = []
    tower_gradients = []
    self.tower_logits_sub = []
    self.tower_mf_auc = []
    self.tower_p_and_n = []

    with tf.device('/cpu:0'):
      self.u = tf.placeholder(tf.int32, [None,]) # [B]
      self.i = tf.placeholder(tf.int32, [None,]) # [B]
      self.j = tf.placeholder(tf.int32, [None,]) # [B]
      self.y = tf.placeholder(tf.float32, [None,]) # [B]
      self.hist_i = tf.placeholder(tf.int32, [None, None]) # [B, T]
      self.sl = tf.placeholder(tf.int32, [None,]) # [B]
      self.lr = tf.placeholder(tf.float64, [])
      u_slices = tf.split(self.u, num_accelerators)
      i_slices = tf.split(self.i, num_accelerators)
      j_slices = tf.split(self.j, num_accelerators)
      y_slices = tf.split(self.y, num_accelerators)
      hist_i_slices = tf.split(self.hist_i, num_accelerators)
      sl_slices = tf.split(self.sl, num_accelerators)

    for device_num, device in enumerate(devices):
      with tf.device(device):
        with tf.variable_scope('GPU_%i' % device_num):
          hidden_units = 128

          user_emb_w = tf.get_variable("user_emb_w", [user_count, hidden_units])
          item_emb_w = tf.get_variable("item_emb_w", [item_count, hidden_units // 2])
          item_b = tf.get_variable("item_b", [item_count],
                               initializer=tf.constant_initializer(0.0))
          cate_emb_w = tf.get_variable("cate_emb_w", [cate_count, hidden_units // 2])
          cate_list = tf.convert_to_tensor(cate_list, dtype=tf.int64)

          ic = tf.gather(cate_list, i_slices[device_num])
          i_emb = tf.concat(values = [
              tf.nn.embedding_lookup(item_emb_w, i_slices[device_num]),
              tf.nn.embedding_lookup(cate_emb_w, ic),
              ], axis=1)
          i_b = tf.gather(item_b, i_slices[device_num])

          jc = tf.gather(cate_list, j_slices[device_num])
          j_emb = tf.concat([
              tf.nn.embedding_lookup(item_emb_w, j_slices[device_num]),
              tf.nn.embedding_lookup(cate_emb_w, jc),
              ], axis=1)
          j_b = tf.gather(item_b, j_slices[device_num])

          hc = tf.gather(cate_list, hist_i_slices[device_num])
          h_emb = tf.concat([
              tf.nn.embedding_lookup(item_emb_w, hist_i_slices[device_num]),
              tf.nn.embedding_lookup(cate_emb_w, hc),
              ], axis=2)

          hist =attention(i_emb, h_emb, sl_slices[device_num])
          #-- attention end ---
          
          hist = tf.layers.batch_normalization(inputs = hist)
          hist = tf.reshape(hist, [-1, hidden_units], name='hist_bn')
          hist = tf.layers.dense(hist, hidden_units, name='hist_fcn')

          u_emb = hist
          #print u_emb.get_shape().as_list()
          #print i_emb.get_shape().as_list()
          #print j_emb.get_shape().as_list()
          #-- fcn begin -------
          din_i = tf.concat([u_emb, i_emb], axis=-1)
          din_i = tf.layers.batch_normalization(inputs=din_i, name='b1')
          d_layer_1_i = tf.layers.dense(din_i, 80, activation=tf.nn.sigmoid, name='f1')
          #if u want try dice change sigmoid to None and add dice layer like following two lines. You can also find model_dice.py in this folder.
          #d_layer_1_i = tf.layers.dense(din_i, 80, activation=None, name='f1')
          #d_layer_1_i = dice(d_layer_1_i, name='dice_1_i')
          d_layer_2_i = tf.layers.dense(d_layer_1_i, 40, activation=tf.nn.sigmoid, name='f2')
          #d_layer_2_i = dice(d_layer_2_i, name='dice_2_i')
          d_layer_3_i = tf.layers.dense(d_layer_2_i, 1, activation=None, name='f3')
          din_j = tf.concat([u_emb, j_emb], axis=-1)
          din_j = tf.layers.batch_normalization(inputs=din_j, name='b1', reuse=True)
          d_layer_1_j = tf.layers.dense(din_j, 80, activation=tf.nn.sigmoid, name='f1', reuse=True)
          #d_layer_1_j = dice(d_layer_1_j, name='dice_1_j')
          d_layer_2_j = tf.layers.dense(d_layer_1_j, 40, activation=tf.nn.sigmoid, name='f2', reuse=True)
          #d_layer_2_j = dice(d_layer_2_j, name='dice_2_j')
          d_layer_3_j = tf.layers.dense(d_layer_2_j, 1, activation=None, name='f3', reuse=True)
          d_layer_3_i = tf.reshape(d_layer_3_i, [-1])
          d_layer_3_j = tf.reshape(d_layer_3_j, [-1])
          x = i_b - j_b + d_layer_3_i - d_layer_3_j # [B]
          logits = i_b + d_layer_3_i
          
          # prediciton for selected items
          # logits for selected item:
          item_emb_all = tf.concat([
              item_emb_w,
              tf.nn.embedding_lookup(cate_emb_w, cate_list)
              ], axis=1)
          item_emb_sub = item_emb_all[:predict_ads_num,:]
          item_emb_sub = tf.expand_dims(item_emb_sub, 0)
          item_emb_sub = tf.tile(item_emb_sub, [predict_batch_size, 1, 1])
          hist_sub =attention_multi_items(item_emb_sub, h_emb, sl_slices[device_num])
          #-- attention end ---
          
          hist_sub = tf.layers.batch_normalization(inputs = hist_sub, name='hist_bn', reuse=tf.AUTO_REUSE)
          # print hist_sub.get_shape().as_list() 
          hist_sub = tf.reshape(hist_sub, [-1, hidden_units])
          hist_sub = tf.layers.dense(hist_sub, hidden_units, name='hist_fcn', reuse=tf.AUTO_REUSE)

          u_emb_sub = hist_sub
          item_emb_sub = tf.reshape(item_emb_sub, [-1, hidden_units])
          din_sub = tf.concat([u_emb_sub, item_emb_sub], axis=-1)
          din_sub = tf.layers.batch_normalization(inputs=din_sub, name='b1', reuse=True)
          d_layer_1_sub = tf.layers.dense(din_sub, 80, activation=tf.nn.sigmoid, name='f1', reuse=True)
          #d_layer_1_sub = dice(d_layer_1_sub, name='dice_1_sub')
          d_layer_2_sub = tf.layers.dense(d_layer_1_sub, 40, activation=tf.nn.sigmoid, name='f2', reuse=True)
          #d_layer_2_sub = dice(d_layer_2_sub, name='dice_2_sub')
          d_layer_3_sub = tf.layers.dense(d_layer_2_sub, 1, activation=None, name='f3', reuse=True)
          d_layer_3_sub = tf.reshape(d_layer_3_sub, [-1, predict_ads_num])
          logits_sub = tf.sigmoid(item_b[:predict_ads_num] + d_layer_3_sub)
          logits_sub = tf.reshape(logits_sub, [-1, predict_ads_num, 1])
          self.tower_logits_sub.append(logits_sub)
          #-- fcn end -------

          
          mf_auc = tf.reduce_mean(tf.to_float(x > 0))
          self.tower_mf_auc.append(mf_auc)
          score_i = tf.sigmoid(i_b + d_layer_3_i)
          score_j = tf.sigmoid(j_b + d_layer_3_j)
          score_i = tf.reshape(score_i, [-1, 1])
          score_j = tf.reshape(score_j, [-1, 1])
          p_and_n = tf.concat([score_i, score_j], axis=-1)
          self.tower_p_and_n.append(p_and_n)
          # print self.p_and_n.get_shape().as_list()

          loss = tf.reduce_mean(
              tf.nn.sigmoid_cross_entropy_with_logits(
                  logits=logits,
                  labels=y_slices[device_num])
              )
          tower_losses.append(loss)

          trainable_params = tf.trainable_variables(scope='GPU_%i' % device_num)
          gradients = tf.gradients(loss, trainable_params)
          tower_gradients.append(gradients)

    with tf.device('/cpu:0'):
      # Step variable
      self.global_step = tf.Variable(0, trainable=False, name='global_step')
      self.global_epoch_step = \
          tf.Variable(0, trainable=False, name='global_epoch_step')
      self.global_epoch_step_op = \
          tf.assign(self.global_epoch_step, self.global_epoch_step+1)

      #self.total_loss = tf.reduce_mean(tower_losses)
      self.total_loss = tower_losses[0]
      # Average gradients
      self.avg_gradients = []
      for tower in zip(*tower_gradients):
        if tower[0] == None:
          self.avg_gradients.append(None)
        else:
          self.avg_gradients.append(tf.reduce_mean(tf.stack(tower), 0))
      clip_gradients, _ = tf.clip_by_global_norm(self.avg_gradients, 5)

    self.train_ops = []
    for device_num, device in enumerate(devices):
      with tf.device(device):
        with tf.variable_scope('GPU_%i' % device_num):
          opt = tf.train.GradientDescentOptimizer(learning_rate=self.lr)
          trainable_params = tf.trainable_variables(scope='GPU_%i' % device_num)
          train_op = opt.apply_gradients(
              zip(clip_gradients, trainable_params), global_step=self.global_step)
          self.train_ops.append(train_op)


  def train(self, sess, uij, l):
    loss, _ = sess.run([self.total_loss, self.train_ops], feed_dict={
        self.u: uij[0],
        self.i: uij[1],
        self.y: uij[2],
        self.hist_i: uij[3],
        self.sl: uij[4],
        self.lr: l,
        })
    return loss

  def eval(self, sess, uij):
    u_auc, socre_p_and_n = sess.run([self.tower_mf_auc[0], self.tower_p_and_n[0]], feed_dict={
        self.u: uij[0],
        self.i: uij[1],
        self.j: uij[2],
        self.hist_i: uij[3],
        self.sl: uij[4],
        })
    return u_auc, socre_p_and_n
  
  def test(self, sess, uij):
    return sess.run(self.tower_logits_sub[0], feed_dict={
        self.u: uij[0],
        self.i: uij[1],
        self.j: uij[2],
        self.hist_i: uij[3],
        self.sl: uij[4],
        })
  

  def save(self, sess, path):
    saver = tf.train.Saver()
    saver.save(sess, save_path=path)

  def restore(self, sess, path):
    saver = tf.train.Saver()
    saver.restore(sess, save_path=path)

def extract_axis_1(data, ind):
  batch_range = tf.range(tf.shape(data)[0])
  indices = tf.stack([batch_range, ind], axis=1)
  res = tf.gather_nd(data, indices)
  return res

def attention(queries, keys, keys_length):
  '''
    queries:     [B, H]
    keys:        [B, T, H]
    keys_length: [B]
  '''
  queries_hidden_units = queries.get_shape().as_list()[-1]
  queries = tf.tile(queries, [1, tf.shape(keys)[1]])
  queries = tf.reshape(queries, [-1, tf.shape(keys)[1], queries_hidden_units])
  din_all = tf.concat([queries, keys, queries-keys, queries*keys], axis=-1)
  d_layer_1_all = tf.layers.dense(din_all, 80, activation=tf.nn.sigmoid, name='f1_att')
  d_layer_2_all = tf.layers.dense(d_layer_1_all, 40, activation=tf.nn.sigmoid, name='f2_att')
  d_layer_3_all = tf.layers.dense(d_layer_2_all, 1, activation=None, name='f3_att')
  d_layer_3_all = tf.reshape(d_layer_3_all, [-1, 1, tf.shape(keys)[1]])
  outputs = d_layer_3_all 
  # Mask
  key_masks = tf.sequence_mask(keys_length, tf.shape(keys)[1])   # [B, T]
  key_masks = tf.expand_dims(key_masks, 1) # [B, 1, T]
  paddings = tf.ones_like(outputs) * (-2 ** 32 + 1)
  outputs = tf.where(key_masks, outputs, paddings)  # [B, 1, T]

  # Scale
  outputs = outputs / (keys.get_shape().as_list()[-1] ** 0.5)

  # Activation
  outputs = tf.nn.softmax(outputs)  # [B, 1, T]

  # Weighted sum
  outputs = tf.matmul(outputs, keys)  # [B, 1, H]

  return outputs

def attention_multi_items(queries, keys, keys_length):
  '''
    queries:     [B, N, H] N is the number of ads
    keys:        [B, T, H] 
    keys_length: [B]
  '''
  queries_hidden_units = queries.get_shape().as_list()[-1]
  queries_nums = queries.get_shape().as_list()[1]
  queries = tf.tile(queries, [1, 1, tf.shape(keys)[1]])
  queries = tf.reshape(queries, [-1, queries_nums, tf.shape(keys)[1], queries_hidden_units]) # shape : [B, N, T, H]
  max_len = tf.shape(keys)[1]
  keys = tf.tile(keys, [1, queries_nums, 1])
  keys = tf.reshape(keys, [-1, queries_nums, max_len, queries_hidden_units]) # shape : [B, N, T, H]
  din_all = tf.concat([queries, keys, queries-keys, queries*keys], axis=-1)
  d_layer_1_all = tf.layers.dense(din_all, 80, activation=tf.nn.sigmoid, name='f1_att', reuse=tf.AUTO_REUSE)
  d_layer_2_all = tf.layers.dense(d_layer_1_all, 40, activation=tf.nn.sigmoid, name='f2_att', reuse=tf.AUTO_REUSE)
  d_layer_3_all = tf.layers.dense(d_layer_2_all, 1, activation=None, name='f3_att', reuse=tf.AUTO_REUSE)
  d_layer_3_all = tf.reshape(d_layer_3_all, [-1, queries_nums, 1, max_len])
  outputs = d_layer_3_all 
  # Mask
  key_masks = tf.sequence_mask(keys_length, max_len)   # [B, T]
  key_masks = tf.tile(key_masks, [1, queries_nums])
  key_masks = tf.reshape(key_masks, [-1, queries_nums, 1, max_len]) # shape : [B, N, 1, T]
  paddings = tf.ones_like(outputs) * (-2 ** 32 + 1)
  outputs = tf.where(key_masks, outputs, paddings)  # [B, N, 1, T]

  # Scale
  outputs = outputs / (keys.get_shape().as_list()[-1] ** 0.5)

  # Activation
  outputs = tf.nn.softmax(outputs)  # [B, N, 1, T]
  outputs = tf.reshape(outputs, [-1, 1, max_len])
  keys = tf.reshape(keys, [-1, max_len, queries_hidden_units])
  #print outputs.get_shape().as_list()
  #print keys.get_sahpe().as_list()
  # Weighted sum
  outputs = tf.matmul(outputs, keys)
  outputs = tf.reshape(outputs, [-1, queries_nums, queries_hidden_units])  # [B, N, 1, H]
  print (outputs.get_shape().as_list())
  return outputs
