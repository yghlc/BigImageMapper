#!/usr/bin/env python
# Filename: hyper_para_ray 
"""
introduction: conduct hyper-parameter testing using ray

authors: Huang Lingcao
email:huanglingcao@gmail.com
add time: 22 February, 2021
"""

from ray import tune

backbones = ['deeplabv3plus_xception65.ini','deeplabv3plus_xception41.ini','deeplabv3plus_xception71.ini',
            'deeplabv3plus_resnet_v1_50_beta.ini','deeplabv3plus_resnet_v1_101_beta.ini',
            'deeplabv3plus_mobilenetv2_coco_voc_trainval.ini','deeplabv3plus_mobilenetv3_large_cityscapes_trainfine.ini',
            'deeplabv3plus_mobilenetv3_small_cityscapes_trainfine.ini','deeplabv3plus_EdgeTPU-DeepLab.ini']



def objective_total_F1(step, alpha, beta):
    # sum F1 score across all regions, all calculate the totaol F1 score.
    return 0.1


def training_function(config,checkpoint_dir=None):
    # Hyperparameters
    alpha, beta = config["alpha"], config["beta"]

    total_F1_score = objective_total_F1(0, alpha, beta)

    # Feed the score back back to Tune.
    tune.report(total_F1=total_F1_score)

# tune.choice([1])  # randomly chose one value

analysis = tune.run(
    training_function,
    config={
        "lr": tune.grid_search([0.0001, 0.007, 0.014, 0.028,0.056]), #0.007, 0.0001,
        "iter_num": tune.grid_search([30000, 60000,90000]),
        "batch_size": tune.grid_search([8, 16, 32, 64, 128]),
        "backbone": tune.grid_search(backbones)
    })

print("Best config: ", analysis.get_best_config(
    metric="total_F1", mode="max"))

# Get a dataframe for analyzing trial results.
df = analysis.results_df
