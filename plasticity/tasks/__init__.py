"""
tasks — task definitions for the plasticity experiments.

    tasks.permuted_pixels     PermutedPixels: a fixed pixel permutation applied to an image dataset
    tasks.toy_regression      Task / make_disjoint_direct_tasks: low-dimensional regression tasks
    tasks.feature_streams     Shapes3D feature-relevance task streams (representation setting)
    tasks.randomized_labels   RandomizedLabelStream: label-randomisation stream (memorization setting)
    tasks.factor_rules        Shapes3D binary tasks cycling through the factor that decides the label

Import from the submodules directly -- toy_regression and feature_streams both
define a class called `Task`.
"""
