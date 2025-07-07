import numpy as np
import tensorflow as tf
from tensorflow import keras
tf.compat.v1.disable_eager_execution()

class ConvNet:
    @staticmethod
    def triple_loss(particles, targets, k, margin):
        loss = 0
        for i in range(k):
            d_p = tf.reduce_mean(tf.square(particles[:, i] - targets[:, i]), 0)
            if i == 0:
                d_n = tf.reduce_mean(tf.square(particles[:, i] - targets[:, i+1]), 0)
            elif i == k-1:
                d_n = tf.reduce_mean(tf.square(particles[:, i] - targets[:, i - 1]), 0)
            else:
                d_n = 0.5 * tf.reduce_mean(tf.square(particles[:, i] - targets[:, i - 1]), 0) + \
                      0.5 * tf.reduce_mean(tf.square(particles[:, i] - targets[:, i - 1]), 0)

            l = tf.maximum(d_p, margin + d_p - d_n)
            loss += l
        return loss / k
    def __init__(self, name=None, folder_name=None, load_path=None,
                 **convnet_pars):
        self._name = name
        self._folder_name = folder_name

        config = tf.compat.v1.ConfigProto()
        config.gpu_options.allow_growth = True

        self._session = tf.compat.v1.Session(config=config)
        if load_path is not None:
            self._load(load_path, convnet_pars)
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

    def predict(self, s, idx=None):
        s = np.transpose(s, [0, 2, 3, 1])
        if idx is not None:
            return self._session.run(self._q[idx], feed_dict={self._x: s})
        else:
            return np.array(
                [self._session.run(self._q, feed_dict={self._x: s})])

    def fit(self, s, a, q, mask, prob_exploration, margin):
        s = np.transpose(s, [0, 2, 3, 1])
        summaries, _ = self._session.run(
            [self._merged, self._train_step],
            feed_dict={self._x: s,
                       self._action: a.ravel().astype(np.uint8),
                       self._target_q: q,
                       self._mask: mask,
                       self._prob_exploration: prob_exploration,
                       self._margin: margin}
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

    def _load(self, path, convnet_pars):
        self._scope_name = 'train/'
        self._folder_name = path
        restorer = tf.train.import_meta_graph(
            path + '/' + self._scope_name[:-1] + '/' + self._scope_name[:-1] +
            '.meta')
        restorer.restore(
            self._session,
            path + '/' + self._scope_name[:-1] + '/' + self._scope_name[:-1]
        )
        self._restore_collection(convnet_pars)

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

            with tf.compat.v1.variable_scope('Mask'):
                self._mask = tf.compat.v1.placeholder(
                    tf.float32, shape=[None, convnet_pars['n_approximators']])

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

                '''def scale_gradient():
                    with flatten.graph.gradient_override_map(
                            {'Identity': 'scaled_gradient_' + self._name}):
                        return tf.identity(flatten, name='identity')

                @tf.RegisterGradient('scaled_gradient_' + self._name)
                def scaled_gradient(op, grad):
                    return grad / float(convnet_pars['n_approximators'])'''

                identity = flatten

            self._features = list()
            self._q = list()
            self._q_acted = list()
            self.n_approximators = convnet_pars['n_approximators']
            self.q_min = convnet_pars['q_min']
            self.q_max = convnet_pars['q_max']
            self.init_type = convnet_pars['init_type']

            if self.init_type == 'boot':
                kernel_initializer = lambda _: tf.compat.v1.glorot_uniform_initializer()
                bias_initializer = lambda _: tf.zeros_initializer()
            else:
                initial_values = np.linspace(self.q_min, self.q_max, self.n_approximators)
                kernel_initializer = lambda _: tf.compat.v1.glorot_uniform_initializer()
                bias_initializer = lambda i: tf.constant_initializer(initial_values[i])

            for i in range(self.n_approximators):
                with tf.compat.v1.variable_scope('head_' + str(i)):
                    self._features.append(tf.compat.v1.layers.dense(
                        identity, 512, activation=tf.nn.relu,
                        kernel_initializer=tf.compat.v1.glorot_uniform_initializer(),
                        name='_features_' + str(i)
                    ))
                    self._q.append(tf.compat.v1.layers.dense(
                        self._features[i],
                        convnet_pars['output_shape'][0],
                        kernel_initializer=kernel_initializer(i),
                        bias_initializer=bias_initializer(i),
                        name='q_' + str(i)
                    ))
                    self._q_acted.append(
                        tf.reduce_sum(self._q[i] * action_one_hot,
                                      axis=1,
                                      name='q_acted_' + str(i))
                    )

            self._q_acted = tf.transpose(self._q_acted)

            self._target_q = tf.compat.v1.placeholder(
                'float32',
                [None, convnet_pars['n_approximators']],
                name='target_q'
            )
            self._margin = tf.compat.v1.placeholder('float32', (),
                                                    name='margin')
            self._q_acted_sorted = tf.sort(self._q_acted, axis=1)
            self._target_q_sorted = tf.sort(self._target_q, axis=1)

            loss = 0.
            if convnet_pars["loss"] == "huber_loss":
                self.loss_fuction = tf.compat.v1.losses.huber_loss
            else:
                self.loss_fuction = tf.compat.v1.losses.mean_squared_error
            k = convnet_pars['n_approximators']
            if convnet_pars["loss"] == "triple_loss":

                loss = ConvNet.triple_loss(self._q_acted_sorted, self._target_q_sorted, k , self._margin)
            else:
                for i in range(convnet_pars['n_approximators']):

                    loss += self.loss_fuction(
                        self._target_q_sorted[:, i],
                        self._q_acted_sorted[:, i]
                    )
                loss = loss / k
            self._prob_exploration = tf.compat.v1.placeholder('float32', (),
                                                    name='prob_exploration')
            tf.compat.v1.summary.scalar(convnet_pars["loss"], loss)
            tf.compat.v1.summary.scalar('average_q', tf.reduce_mean(self._q))
            tf.compat.v1.summary.scalar('prob_exploration', self._prob_exploration)
            tf.compat.v1.summary.histogram('qs', self._q)
            self._merged = tf.compat.v1.summary.merge_all()
            optimizer = convnet_pars['optimizer']
            if optimizer['name'] == 'rmspropcentered':
                opt = tf.keras.optimizers.RMSprop(learning_rate=optimizer['lr'],
                                                decay=optimizer['decay'],
                                                epsilon=optimizer['epsilon'],
                                                centered=True)
            elif optimizer['name'] == 'rmsprop':
                opt = tf.compat.v1.train.RMSPropOptimizer(learning_rate=optimizer['lr'],
                                                decay=optimizer['decay'],
                                                epsilon=optimizer['epsilon'])
            elif optimizer['name'] == 'adam':
                opt = tf.compat.v1.train.AdamOptimizer(learning_rate=optimizer['lr'])
            elif optimizer['name'] == 'adadelta':
                opt = tf.compat.v1.train.AdadeltaOptimizer(learning_rate=optimizer['lr'])
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

    def _add_collection(self):
        tf.compat.v1.add_to_collection(self._scope_name + '_x', self._x)
        tf.compat.v1.add_to_collection(self._scope_name + '_action', self._action)

        for i in range(len(self._features)):
            tf.compat.v1.add_to_collection(self._scope_name + '_features_' + str(i),
                                 self._features[i])
            tf.compat.v1.add_to_collection(self._scope_name + '_q_' + str(i), self._q[i])
            tf.compat.v1.add_to_collection(self._scope_name + '_q_acted_' + str(i),
                                 self._q_acted[i])
        tf.compat.v1.add_to_collection(self._scope_name + '_target_q', self._target_q)
        tf.compat.v1.add_to_collection(self._scope_name + '_merged', self._merged)
        tf.compat.v1.add_to_collection(self._scope_name + '_train_step', self._train_step)
        tf.compat.v1.add_to_collection(self._scope_name + '_mask', self._mask)

    def _restore_collection(self, convnet_pars):
        self._x = tf.compat.v1.get_collection(self._scope_name + '_x')[0]
        self._action = tf.compat.v1.get_collection(self._scope_name + '_action')[0]

        features = list()
        q = list()
        q_acted = list()
        for i in range(convnet_pars['n_approximators']):
            features.append(tf.compat.v1.get_collection(
                self._scope_name + '_features_' + str(i))[0])
            q.append(tf.compat.v1.get_collection(self._scope_name + '_q_' + str(i))[0])
            q_acted.append(tf.compat.v1.get_collection(
                self._scope_name + '_q_acted_' + str(i))[0])

        self._features = features
        self._q = q
        self._q_acted = q_acted
        self._target_q = tf.compat.v1.get_collection(self._scope_name + '_target_q')[0]
        self._merged = tf.compat.v1.get_collection(self._scope_name + '_merged')[0]
        self._train_step = tf.compat.v1.get_collection(
            self._scope_name + '_train_step')[0]

        self._mask = tf.compat.v1.placeholder(
            tf.float32, shape=[None, convnet_pars['n_approximators']])

        self._train_count = 0