import numpy as np
import tensorflow as tf
tf.compat.v1.disable_eager_execution()

class ConvNet:
    def __init__(self, name=None, folder_name=None, load_path=None,
                 **convnet_pars):
        self._name = name
        self._folder_name = folder_name

        config = tf.compat.v1.ConfigProto()
        config.gpu_options.allow_growth = True

        self._session = tf.compat.v1.Session(config=config)

        if load_path is not None:
            self._load(load_path)
        else:
            self._build(convnet_pars)

        if self._name == 'train':
            self._train_saver = tf.compat.v1.train.Saver(
                tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES,
                                  scope=self._scope_name))
        elif self._name == 'target':
            w = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES,
                                  scope=self._scope_name)

            with tf.compat.v1.variable_scope(self._scope_name):
                self._target_w = list()
                self._w = list()
                with tf.compat.v1.variable_scope('weights_placeholder'):
                    for i in range(len(w)):
                        self._target_w.append(tf.compat.v1.placeholder(w[i].dtype,
                                                             shape=w[i].shape))
                        self._w.append(w[i].assign(self._target_w[i]))

    def predict(self, s, features=False):
        s = np.transpose(s, [0, 2, 3, 1])
        if not features:
            return self._session.run(self.q, feed_dict={self._x: s})
        else:
            return self._session.run(self._features, feed_dict={self._x: s})

    def fit(self, s, a, q):
        s = np.transpose(s, [0, 2, 3, 1])
        summaries, _ = self._session.run(
            [self._merged, self._train_step],
            feed_dict={self._x: s,
                       self._action: a.ravel().astype(np.uint8),
                       self._target_q: q}
        )
        if hasattr(self, '_train_writer'):
            self._train_writer.add_summary(summaries, self._train_count)

        self._train_count += 1

    def set_weights(self, weights):
        w = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES,
                              scope=self._scope_name)
        assert len(w) == len(weights)

        for i in range(len(w)):
            self._session.run(self._w[i],
                              feed_dict={self._target_w[i]: weights[i]})

    def get_weights(self, only_trainable=False):
        if not only_trainable:
            w = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES,
                                  scope=self._scope_name)
        else:
            w = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES,
                                  scope=self._scope_name)

        return self._session.run(w)

    def save(self):
        self._train_saver.save(
            self._session,
            self._folder_name + '/' + self._scope_name[:-1] + '/' +
            self._scope_name[:-1]
        )

    def _load(self, path):
        self._scope_name = 'train/'
        restorer = tf.train.import_meta_graph(
            path + '/' + self._scope_name[:-1] + '/' + self._scope_name[:-1] +
            '.meta')
        restorer.restore(
            self._session,
            path + '/' + self._scope_name[:-1] + '/' + self._scope_name[:-1]
        )
        self._restore_collection()

    def _build(self, convnet_pars):

        with tf.compat.v1.variable_scope(None, default_name=self._name):
            self._scope_name = tf.compat.v1.get_default_graph().get_name_scope() + '/'
            with tf.compat.v1.variable_scope('State'):
                self._x = tf.compat.v1.placeholder(tf.float32,
                                         shape=[None] + list(
                                             convnet_pars['input_shape']),
                                         name='input')

            with tf.compat.v1.variable_scope('Action'):
                self._action = tf.compat.v1.placeholder('uint8', [None], name='action')

                action_one_hot = tf.one_hot(self._action,
                                            convnet_pars['output_shape'][0],
                                            name='action_one_hot')

            with tf.compat.v1.variable_scope('Convolutions'):
                hidden_1 = tf.compat.v1.layers.conv2d(
                    self._x / 255., 32, 8, 4, activation=tf.nn.relu,
                    kernel_initializer=tf.compat.v1.glorot_uniform_initializer(),
                    name='hidden_1'
                )
                hidden_2 = tf.compat.v1.layers.conv2d(
                    hidden_1, 64, 4, 2, activation=tf.nn.relu,
                    kernel_initializer=tf.compat.v1.glorot_uniform_initializer(),
                    name='hidden_2'
                )
                hidden_3 = tf.compat.v1.layers.conv2d(
                    hidden_2, 64, 3, 1, activation=tf.nn.relu,
                    kernel_initializer=tf.compat.v1.glorot_uniform_initializer(),
                    name='hidden_3'
                )
                flatten = tf.reshape(hidden_3, [-1, 7 * 7 * 64], name='flatten')

            self._features = tf.compat.v1.layers.dense(
                flatten, 512, activation=tf.nn.relu,
                kernel_initializer=tf.compat.v1.glorot_uniform_initializer(),
                bias_initializer=tf.compat.v1.glorot_uniform_initializer(),
                name='_features'
            )
            self.q = tf.compat.v1.layers.dense(
                self._features, convnet_pars['output_shape'][0],
                kernel_initializer=tf.compat.v1.glorot_uniform_initializer(),
                bias_initializer=tf.compat.v1.glorot_uniform_initializer(),
                name='q'
            )

            self._target_q = tf.compat.v1.placeholder('float32', [None], name='target_q')
            self._q_acted = tf.reduce_sum(self.q * action_one_hot,
                                          axis=1,
                                          name='q_acted')

            loss = tf.compat.v1.losses.huber_loss(self._target_q, self._q_acted)
            tf.compat.v1.summary.scalar('huber_loss', loss)
            tf.compat.v1.summary.scalar('average_q', tf.reduce_mean(self.q))
            self._merged = tf.compat.v1.summary.merge_all()

            optimizer = convnet_pars['optimizer']
            if optimizer['name'] == 'rmspropcentered':
                opt = tf.train.RMSPropOptimizer(learning_rate=optimizer['lr'],
                                                decay=optimizer['decay'],
                                                epsilon=optimizer['epsilon'],
                                                centered=True)
            elif optimizer['name'] == 'rmsprop':
                opt = tf.compat.v1.train.RMSPropOptimizer(learning_rate=optimizer['lr'],
                                                decay=optimizer['decay'],
                                                epsilon=optimizer['epsilon'])
            elif optimizer['name'] == 'adam':
                opt = tf.train.AdamOptimizer(learning_rate=optimizer['lr'])
            elif optimizer['name'] == 'adadelta':
                opt = tf.train.AdadeltaOptimizer(learning_rate=optimizer['lr'])
            else:
                raise ValueError('Unavailable optimizer selected.')

            self._train_step = opt.minimize(loss=loss)

            initializer = tf.compat.v1.variables_initializer(
                tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES,
                                  scope=self._scope_name))

        self._session.run(initializer)

        if self._folder_name is not None:
            self._train_writer = tf.compat.v1.summary.FileWriter(
                self._folder_name + '/' + self._scope_name[:-1],
                graph=tf.compat.v1.get_default_graph()
            )

        self._train_count = 0

        self._add_collection()

    @property
    def n_features(self):
        return self._features.shape[-1]

    def _add_collection(self):
        tf.compat.v1.add_to_collection(self._scope_name + '_x', self._x)
        tf.compat.v1.add_to_collection(self._scope_name + '_action', self._action)
        tf.compat.v1.add_to_collection(self._scope_name + '_features', self._features)
        tf.compat.v1.add_to_collection(self._scope_name + '_q', self.q)
        tf.compat.v1.add_to_collection(self._scope_name + '_target_q', self._target_q)
        tf.compat.v1.add_to_collection(self._scope_name + '_q_acted', self._q_acted)
        tf.compat.v1.add_to_collection(self._scope_name + '_merged', self._merged)
        tf.compat.v1.add_to_collection(self._scope_name + '_train_step', self._train_step)

    def _restore_collection(self):
        self._x = tf.compat.v1.get_collection(self._scope_name + '_x')[0]
        self._action = tf.compat.v1.get_collection(self._scope_name + '_action')[0]
        self._features = tf.compat.v1.get_collection(self._scope_name + '_features')[0]
        self.q = tf.compat.v1.get_collection(self._scope_name + '_q')[0]
        self._target_q = tf.compat.v1.get_collection(self._scope_name + '_target_q')[0]
        self._q_acted = tf.compat.v1.get_collection(self._scope_name + '_q_acted')[0]
        self._merged = tf.compat.v1.get_collection(self._scope_name + '_merged')[0]
        self._train_step = tf.compat.v1.get_collection(self._scope_name + '_train_step')[0]