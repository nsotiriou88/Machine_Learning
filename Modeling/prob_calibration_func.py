#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 10:41:37 2020

@author: Nikolaos Sotiriou - github: nsotiriou88 - email: nsotiriou88@gmail.com
"""


# Probability calibration function (input: data array of predictors probability results)
def calibration(data, train_pop, target_pop, sampled_train_pop, sampled_target_pop):
    '''
    Calibration function to update probability distribution over an undersampled sample.
    
    link: https://towardsdatascience.com/how-to-calibrate-undersampled-model-scores-8f3319c1ea5b
    '''
    calibrated_data = \
    ((data * (target_pop / train_pop) / (sampled_target_pop / sampled_train_pop)) /
    ((
        (1 - data) * (1 - target_pop / train_pop) / (1 - sampled_target_pop / sampled_train_pop)
     ) +
     (
        data * (target_pop / train_pop) / (sampled_target_pop / sampled_train_pop)
     )))

    return calibrated_data
