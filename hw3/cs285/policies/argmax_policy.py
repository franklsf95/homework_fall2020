import numpy as np


class ArgMaxPolicy(object):
    def __init__(self, critic):
        self.critic = critic

    def get_action(self, obs):
        if len(obs.shape) > 3:
            observation = obs
        else:
            observation = obs[None]

        ## return the action that maximizes the Q-value
        # at the current observation as the output
        qa_t_values = self.critic.qa_values(observation)
        action = np.argmax(qa_t_values, axis=1)

        return action.squeeze()