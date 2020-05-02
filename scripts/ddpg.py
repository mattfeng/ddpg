#!/usr/bin/env python

import torch
import torch.nn as nn
import numpy as np

class ValueModel(nn.Module):
    def __init__(self, state_dim, action_dim, value_dim=1):
        super(ValueModel, self).__init__()

        self.value_dim = value_dim
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.sa_dim = state_dim + action_dim
        self.f1 = nn.Linear(self.sa_dim, 32)
        self.f2 = nn.Linear(32, self.value_dim)
    
    def process_state(self, state):
        """Turns a state into a flat vector."""
        return state
    
    def process_action(self, action):
        """Turns an action into a flat vector."""
        if len(action.shape) != 2:
            return action.reshape(-1, self.action_dim)
        return action
    
    def forward(self, state_in, action_in):
        # ? means variable number of observations
        state = self.process_state(state_in)     # (? x s_dim)
        action = self.process_action(action_in)  # (? x a_dim)
        sa = torch.cat([state, action], dim=1)   # (? x (s_dim + a_dim))

        value = torch.relu(self.f1(sa))
        value = self.f2(value)

        return value


class PolicyModel(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(PolicyModel, self).__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.f1 = nn.Linear(self.state_dim, 128)
        self.f2 = nn.Linear(128, 128)
        self.f3 = nn.Linear(128, self.action_dim)
        
    def process_state(self, state):
        return state

    def forward(self, state_in):
        state = self.process_state(state_in)
        action = torch.relu(self.f1(state))
        action = torch.relu(self.f2(action))
        action = self.f3(action)

        return action

def bellman_loss(
        sarsd_arr,
        tgt_val_net,
        tgt_pol_net,
        opt_val_net,
        discount):
    """
    sarsd_arr: Array of 5 arrays s, a, r, sp, d.
    """
    s, a, r, sp, d = [torch.from_numpy(np.array(x, dtype=np.float32)) for x in sarsd_arr]
    tgt_actions = tgt_pol_net(sp)
    # a better prediction of Q(s, a) = E[r + gamma * max[a']Q(s', a')]
    tgt_val = r + discount * (1 - d) * tgt_val_net(sp, tgt_actions)
    cur_val = opt_val_net(s, a)

    return torch.mean((tgt_val - cur_val) ** 2)

def policy_loss(sarsd_arr, opt_val_net, opt_pol_net):
    s, _, _, _, _ = [torch.from_numpy(np.array(x, dtype=np.float32)) for x in sarsd_arr]
    return -torch.mean(opt_val_net(s, opt_pol_net(s)))