# Copyright Axelera AI, 2023
from axelera.app import operators


class Op(operators.AxOperator):
    pass


class PermuteChannels(operators.AxOperator):
    '''Note overrides the builtin PermuteChannels operator'''
