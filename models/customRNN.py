import tensorflow as tf
import ipdb
import logging

log = logging.getLogger('root.customrnn')

class LocationDecoder(object):
    def __init__(self, encoder, labels, maxtimesteps=10, init_loc=None):
        self.encoderout = encoder
        self.labels = labels
        self.maxtimesteps = maxtimesteps
        self.cell = LocationPredictorCell(name='locationcell', whh=encoder,
                                          location_dimension=labels.shape[-1])
        self.init_loc = tf.tile(tf.expand_dims(tf.concat([init_loc, tf.constant([0], dtype=tf.float32)], axis=0),
                                                         axis=0),
                                               [tf.shape(encoder)[0], 1])
        #ipdb.set_trace()
        self.decoderunits = encoder.get_shape()[1]

    def step(self, prevstate, prevloc):
        #log.info('state-{}, loc-{}'.format(prevstate.get_shape(), prevloc.get_shape()))
        if self.cell.built is False:
            self.cell.build(tf.shape(prevloc))

        new_loc, new_h = self.cell.call(prevstate, prevloc)
        return new_loc, new_h

    def unroll(self):
        all_hidden = []
        all_loc = []
        hidden = tf.zeros(shape=[tf.shape(self.encoderout)[0], self.decoderunits.value], dtype=tf.float32)
        loc = self.init_loc
        for i in range(self.maxtimesteps):
            loc, hidden = self.step(hidden, loc)
            all_hidden.append(hidden)
            all_loc.append(loc)

        loc = tf.stack(all_loc, axis=1)
        hidden = tf.stack(all_hidden, axis=1)

        return loc, hidden


class LocationPredictorCell(object):
    """
    Args:
      num_units: int, The number of units in the GRU cell.
      activation: Nonlinearity to use.  Default: `tanh`.
      reuse: (optional) Python boolean describing whether to reuse variables
       in an existing scope.  If not `True`, and the existing scope already has
       the given variables, an error is raised.
      kernel_initializer: (optional) The initializer to use for the weight and
      projection matrices.
      bias_initializer: (optional) The initializer to use for the bias.
      name: String, the name of the layer. Layers with the same name will
        share weights, but to avoid mistakes we require reuse=True in such
        cases.
    """

    def __init__(self,
                 #num_units,
                 activation=None,
                 reuse=None,
                 kernel_initializer=None,
                 bias_initializer=None,
                 name=None,
                 whh=None,
                 location_dimension=None):
        self._num_units = whh.get_shape()[1]
        self.whh = whh
        self.loc_dim = location_dimension
        self.activation = activation or tf.nn.elu
        self.built = False
        #assert self.whh.get_shape()[-1] == num_units, 'cannot create custom rnn cell'

    @property
    def state_size(self):
        return self._num_units

    @property
    def output_size(self):
        return self._num_units

    def build(self, inputs_shape):

        if hasattr(tf, 'initializers'):
            initializer = tf.initializers.random_uniform
        else:
            initializer = tf.random_uniform_initializer()

        #self._Umat = tf.Variable(tf.random_normal([inputs_shape[1], self._num_units]),
        # 126 x 126
        self._Umat = tf.get_variable('ipkernel_{}'.format(self.__class__.__name__),
                                     shape=([self._num_units, self._num_units]), initializer=initializer)

        # 126
        self._bias = tf.get_variable('bias_{}'.format(self.__class__.__name__), initializer=initializer,
                                    shape=[self._num_units])

        # 126x65
        self._locw = tf.get_variable('locw', shape=[self._num_units, self.loc_dim + 1])
        # 64x8
        self._glimpsetransform = tf.get_variable('glimpseT', shape=[self.loc_dim, self.whh.get_shape()[2]])
        # 8
        self._glbias = tf.get_variable('glbias', shape=[self.whh.get_shape()[2]])
        self.built = True

    def call(self, state, prevloc):
        """Gated recurrent unit (GRU) with nunits cells."""
        #ipdb.set_trace()
        glimpse_attention = tf.nn.sigmoid(tf.matmul(prevloc[:, :-1], self._glimpsetransform) + self._glbias)
        new_glimpse = tf.reduce_sum(tf.multiply(self.whh, tf.expand_dims(glimpse_attention, axis=1)), axis=2)

        new_h = tf.nn.elu(new_glimpse + tf.matmul(state, self._Umat) + self._bias)
        new_loc = self.activation(tf.matmul(new_h, self._locw))
        return new_loc, new_h
