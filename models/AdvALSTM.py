import tensorflow as tf
from experiments.exp_lstms import Exp_LSTMs
class LSTMs(Exp_LSTMs):
    def __init__(self, father):
        self.father = father

    def construct_graph(self):
        print('is pred_lstm')
        if self.father.gpu != -1:
            device_name = f'/gpu:{self.father.gpu}'
        else:
            device_name = '/cpu:0'
        print('device name:', device_name)
        with tf.device(device_name):
            tf.reset_default_graph()
            if self.father.fix_init:
                tf.set_random_seed(123456)

            self.father.gt_var = tf.placeholder(tf.float32, [None, 1])
            self.father.pv_var = tf.placeholder(
                tf.float32, [None, self.father.paras['seq'], self.father.fea_dim]
            )
            self.father.wd_var = tf.placeholder(
                tf.float32, [None, self.father.paras['seq'], 5]
            )

            self.father.lstm_cell = tf.contrib.rnn.BasicLSTMCell(
                self.father.paras['unit']
            )

            # self.outputs, _ = tf.nn.dynamic_rnn(
            #     # self.outputs, _ = tf.nn.static_rnn(
            #     self.lstm_cell, self.pv_var, dtype=tf.float32
            #     # , initial_state=ini_sta
            # )

            self.father.in_lat = tf.layers.dense(
                self.father.pv_var, units=self.father.fea_dim,
                activation=tf.nn.tanh, name='in_fc',
                kernel_initializer=tf.glorot_uniform_initializer()
            )

            self.father.outputs, _ = tf.nn.dynamic_rnn(
                # self.father.outputs, _ = tf.nn.static_rnn(
                self.father.lstm_cell, self.father.in_lat, dtype=tf.float32
                # , initial_state=ini_sta
            )

            self.father.loss = 0
            self.father.adv_loss = 0
            self.father.l2_norm = 0
            if self.father.att:
                with tf.variable_scope('lstm_att') as scope:
                    self.father.av_W = tf.get_variable(
                        name='att_W', dtype=tf.float32,
                        shape=[self.father.paras['unit'], self.father.paras['unit']],
                        initializer=tf.glorot_uniform_initializer()
                    )
                    self.father.av_b = tf.get_variable(
                        name='att_h', dtype=tf.float32,
                        shape=[self.father.paras['unit']],
                        initializer=tf.zeros_initializer()
                    )
                    self.father.av_u = tf.get_variable(
                        name='att_u', dtype=tf.float32,
                        shape=[self.father.paras['unit']],
                        initializer=tf.glorot_uniform_initializer()
                    )

                    self.father.a_laten = tf.tanh(
                        tf.tensordot(self.father.outputs, self.father.av_W,
                                     axes=1) + self.father.av_b)
                    self.father.a_scores = tf.tensordot(self.father.a_laten, self.father.av_u,
                                                 axes=1,
                                                 name='scores')
                    self.father.a_alphas = tf.nn.softmax(self.father.a_scores, name='alphas')

                    self.father.a_con = tf.reduce_sum(
                        self.father.outputs * tf.expand_dims(self.father.a_alphas, -1), 1)
                    self.father.fea_con = tf.concat(
                        [self.father.outputs[:, -1, :], self.father.a_con],
                        axis=1)
                    print('adversarial scope')
                    # training loss
                    self.father.pred = self.father.adv_part(self.father.fea_con)
                    if self.father.hinge:
                        self.father.loss = tf.losses.hinge_loss(self.father.gt_var, self.father.pred)
                    else:
                        self.father.loss = tf.losses.log_loss(self.father.gt_var, self.father.pred)

                    self.father.adv_loss = self.father.loss * 0

                    # adversarial loss
                    if self.father.adv_train:
                        print('gradient noise')
                        self.father.delta_adv = tf.gradients(self.father.loss, [self.father.fea_con])[0]
                        tf.stop_gradient(self.father.delta_adv)
                        self.father.delta_adv = tf.nn.l2_normalize(self.father.delta_adv, axis=1)
                        self.father.adv_pv_var = self.father.fea_con + \
                                          self.father.paras['eps'] * self.father.delta_adv

                        scope.reuse_variables()
                        self.father.adv_pred = self.father.adv_part(self.father.adv_pv_var)
                        if self.father.hinge:
                            self.father.adv_loss = tf.losses.hinge_loss(self.father.gt_var, self.father.adv_pred)
                        else:
                            self.father.adv_loss = tf.losses.log_loss(self.father.gt_var, self.father.adv_pred)
            else:
                with tf.variable_scope('lstm_att') as scope:
                    print('adversarial scope')
                    # training loss
                    self.father.pred = self.father.adv_part(self.father.outputs[:, -1, :])
                    if self.father.hinge:
                        self.father.loss = tf.losses.hinge_loss(self.father.gt_var, self.father.pred) #铰链损失函数(SVM中使用）
                    else:
                        self.father.loss = tf.losses.log_loss(self.father.gt_var, self.father.pred) #交叉熵

                    self.father.adv_loss = self.father.loss * 0

                    # adversarial loss
                    if self.father.adv_train:
                        print('gradient noise')
                        self.father.delta_adv = tf.gradients(self.father.loss, [self.father.outputs[:, -1, :]])[0]
                        tf.stop_gradient(self.father.delta_adv)
                        self.father.delta_adv = tf.nn.l2_normalize(self.father.delta_adv,
                                                            axis=1)
                        self.father.adv_pv_var = self.father.outputs[:, -1, :] + \
                                          self.father.paras['eps'] * self.father.delta_adv

                        scope.reuse_variables()
                        self.father.adv_pred = self.father.adv_part(self.father.adv_pv_var)
                        if self.father.hinge:
                            self.father.adv_loss = tf.losses.hinge_loss(self.father.gt_var,
                                                                 self.father.adv_pred)
                        else:
                            self.father.adv_loss = tf.losses.log_loss(self.father.gt_var,
                                                               self.father.adv_pred)

            # regularizer
            self.father.tra_vars = tf.trainable_variables('lstm_att/pre_fc')
            for var in self.father.tra_vars:
                self.father.l2_norm += tf.nn.l2_loss(var)

            self.father.obj_func = self.father.loss + \
                            self.father.paras['alp'] * self.father.l2_norm + \
                            self.father.paras['bet'] * self.father.adv_loss

            self.father.optimizer = tf.train.AdamOptimizer(
                learning_rate=self.father.paras['lr']
            ).minimize(self.father.obj_func)