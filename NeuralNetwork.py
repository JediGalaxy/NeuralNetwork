# -*- coding: utf-8 -*-
"""
Created on Thu Mar  4 13:21:06 2021

@author: Alexander_Maltsev
"""

import tensorflow as tf
import numpy as np

# Политика в форме нейронной сети
#------------------------------------------------------------------------------
def reset_graph(seed=42):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)

def calculation():
    n_inputs = 4
    n_hidden = 4
    n_outputs = 1
    learning_rate = 0.01
    reset_graph()
    initializer = tf.contrib.layers.variance_scaling_initializer()


    X = tf.placeholder(tf.float32, shape=[None, n_inputs])
    hidden = tf.layers.dense(X, n_hidden, activation=tf.nn.elu,
                             kernel_initializer=initializer)
    logits = tf.layers.dense(hidden, n_outputs,
                             kernel_initializer=initializer)
    outputs = tf.nn.sigmoid(logits)

    p_left_and_right = tf.concat(axis=1, values=[outputs, 1 - outputs])
    action = tf.multinomial(tf.log(p_left_and_right), num_samples=1)

    y = 1. - tf.to_float(action)
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y,
                                                            logits=logits)
    optimizer = tf.train.AdamOptimizer(learning_rate)
    grads_and_vars = optimizer.compute_gradients(cross_entropy) # Здесь и здесь
    gradients = [grad for grad, variable in grads_and_vars]     # В массив добавляются
                                                                # элементы NoneType
                                                                # выяснить причину (Ответ в функции def reset_graph())
    gradient_placeholders = []
    grads_and_vars_feed = []

    for grad, variable in grads_and_vars:
        gradient_placeholder = tf.placeholder(tf.float32,
                                              shape=grad.get_shape())
        gradient_placeholders.append(gradient_placeholder)
        grads_and_vars_feed.append((gradient_placeholder, variable))
    training_op = optimizer.apply_gradients(grads_and_vars_feed)

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
#------------------------------------------------------------------------------
def discount_rewards(rewards, discount_rate):
    discounted_rewards = np.zeros(len(rewards))
    cumulative_rewards = 0
    for step in reversed(range(len(rewards))):
        cumulative_rewards = rewards[step] + cumulative_rewards * discount_rate
        discounted_rewards[step] = cumulative_rewards
    return discounted_rewards

def discount_and_normalize_rewards(all_rewards, discount_rate):
    all_discounted_rewards = [discount_rewards(rewards, discount_rate) for rewards in all_rewards]
    flat_rewards = np.concatenate(all_discounted_rewards)
    reward_mean = flat_rewards.mean()
    reward_std = flat_rewards.std()
    return [(discounted_rewards - reward_mean)/reward_std for discounted_rewards in all_discounted_rewards]
#------------------------------------------------------------------------------
#discount_rewards([10, 0, -50], discount_rate=0.8)
#------------------------------------------------------------------------------
#discount_and_normalize_rewards([[10, 0, -50], [10, 20]], discount_rate=0.8)

