# -*- coding: utf-8 -*-
"""
Created on Tue Mar  2 14:02:00 2021

@author: Alexander_Maltsev
"""

class Policy:
    def __init__(self):
        return
    
    def basic_policy(self, obs):
        angle = obs[2]
        return 0 if angle < 0 else 1
    