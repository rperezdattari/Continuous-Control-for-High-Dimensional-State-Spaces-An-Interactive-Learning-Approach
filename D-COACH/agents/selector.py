from agents.HD_agent_ISER import Agent as HD_Basic
from agents.HD_agent_ICRA import Agent as HD_Enhanced
from agents.LD_agent import Agent as LD
from agents.LD_lstm_agent import Agent as  LD_lstm


#  Select agent
def agent_selector(network, version, train_ae, load_policy, learning_rate, dim_a, fc_layers_neurons, loss_function_type,
                   policy_loc, action_upper_limits, action_lower_limits, e, config_graph, config_general):

    if network == 'HD':
        if version == 'Basic':
            return HD_Basic(train_ae=train_ae, load_policy=load_policy, learning_rate=learning_rate, dim_a=dim_a,
                            fc_layers_neurons=fc_layers_neurons, loss_function_type=loss_function_type, policy_loc=policy_loc,
                            action_upper_limits=action_upper_limits, action_lower_limits=action_lower_limits, e=e,
                            ae_loc=config_graph['ae_loc'], image_size=config_graph.getint('image_side_length'),
                            show_ae_output=config_general.getboolean('show_ae_output'),
                            show_state=config_general.getboolean('show_state'),
                            resize_observation=config_general.getboolean('resize_observation'))

        elif version == 'Enhanced':
            return HD_Enhanced(load_policy=load_policy, learning_rate=learning_rate, dim_a=dim_a,
                               fc_layers_neurons=fc_layers_neurons, loss_function_type=loss_function_type,
                               policy_loc=policy_loc, action_upper_limits=action_upper_limits,
                               action_lower_limits=action_lower_limits, e=e,
                               image_size=config_graph.getint('image_side_length'),
                               show_ae_output=config_general.getboolean('show_ae_output'),
                               show_state=config_general.getboolean('show_state'),
                               resize_observation=config_general.getboolean('resize_observation'),
                               ae_training_threshold=float(config_graph['ae_training_threshold']),
                               ae_evaluation_frequency=config_graph.getint('ae_evaluation_frequency'))

    else:
        raise NameError('Not valid network.')
