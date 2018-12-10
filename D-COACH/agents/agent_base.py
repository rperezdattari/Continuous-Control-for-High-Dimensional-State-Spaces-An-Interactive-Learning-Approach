import tensorflow as tf
import numpy as np
from tools.functions import str_2_array
import os


class AgentBase:
    def __init__(self, dim_a=3, policy_loc='./racing_car_m2/network',
                 action_upper_limits='1,1', action_lower_limits='-1,-1',
                 e='1', load_policy=False, **kwargs):

        # Initialize variables
        self.high_dim_observation = None
        self.low_dim_observation = None
        self.network_input = None
        self.y_label = None
        self.e = np.array(str_2_array(e, type_n='float'))
        self.dim_a = dim_a

        self.policy_loc = policy_loc + 'network'

        self.action_upper_limits = str_2_array(action_upper_limits)
        self.action_lower_limits = str_2_array(action_lower_limits)

        # Build and load network if requested
        self._build_network(dim_a, kwargs)

        if load_policy:
            self._load_network()

    def _build_network(self, *args):
        with tf.variable_scope('base'):
            self.y = None
            self.low_dim_input_shape = None
        self.train_policy = None
        self.sess = None
        self.saver = None

        print('\nNetwork builder not implemented!\n')
        exit()

    def _load_network(self):
            self.saver.restore(self.sess, self.policy_loc)

    def _preprocess_observation(self, observation):
        pass

    def update(self, h, observation):
        self._preprocess_observation(observation)

        action = self.y.eval(session=self.sess, feed_dict={'base/input:0': self.network_input})

        error = np.array(h * self.e).reshape(1, self.dim_a)
        self.y_label = []

        for i in range(self.dim_a):
            self.y_label.append(np.clip(action[0, i] + error[0, i],
                                        self.action_lower_limits[i],
                                        self.action_upper_limits[i]))

        self.y_label = np.array(self.y_label).reshape(1, self.dim_a)

        self.sess.run(self.train_policy, feed_dict={'base/input:0': self.network_input,
                                                  'base/label:0': self.y_label})

    def batch_update(self, batch):
        state_batch = [np.array(pair[0]) for pair in batch]
        y_label_batch = [np.array(pair[1]) for pair in batch]

        self._batch_update_extra(state_batch, y_label_batch)

        self.sess.run(self.train_policy, feed_dict={'base/input:0': state_batch,
                                                  'base/label:0': y_label_batch})

    def action(self, observation):
        self._preprocess_observation(observation)

        action = self.y.eval(session=self.sess, feed_dict={'base/input:0': self.network_input})
        out_action = []

        for i in range(self.dim_a):
            action[0, i] = np.clip(action[0, i], self.action_lower_limits[i], self.action_upper_limits[i])
            out_action.append(action[0, i])

        return np.array(out_action)

    def last_step(self):
        return [np.squeeze(self.network_input, axis=0), self.y_label.reshape(self.dim_a)]

    def save_params(self):
        if not os.path.exists(self.policy_loc):
            os.makedirs(self.policy_loc)

        self.saver.save(self.sess, self.policy_loc)

    def _batch_update_extra(self, state_batch, y_label_batch):
        pass

    def time_step(self, t):
        pass

    def new_episode(self):
        pass
