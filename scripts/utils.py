#!/usr/bin/env python

import random
from collections import deque

class ReplayBuffer(object):
    def __init__(self, buf_sz):
        self.buf_sz = buf_sz
        self.buffer = deque(maxlen=buf_sz)

    def add(self, sarsd_arr):
        self.buffer.extend(sarsd_arr)
    
    def add_one(self, sarsd):
        self.buffer.append(sarsd)

    def get_tuples(self, batch_sz=None):
        """Returns (s,a,r,s',d) tuples of experience."""
        if batch_sz is None:
            return self.buffer
        
        return random.sample(self.buffer, batch_sz)
    
    def get_arrays(self, batch_sz=None):
        """Returns five arrays corresponding to (s,a,r,s',d)."""
        if batch_sz is None:
            return zip(*self.buffer)

        sample = random.sample(self.buffer, batch_sz)
        return zip(*sample)

class RunningAverage(object):
    def __init__(self, memory):
        self.memory = memory
        self.values = deque(maxlen=memory)

    def update(self, val):
        self.values.append(val)

    def avg(self):
        return sum(self.values) / len(self.values)