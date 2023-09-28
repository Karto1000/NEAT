# -----------------------------------------------------------
# Copyright (c) YPSOMED AG, Burgdorf / Switzerland
# YDS INNOVATION - Digital Innovation
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# email diginno@ypsomed.com
# author: Tim Leuenberger (Tim.leuenberger@ypsomed.com)
# -----------------------------------------------------------
from enum import Enum


class NodeType(Enum):
    INPUT = 1
    HIDDEN = 2
    OUTPUT = 3


class GeneNode:
    def __init__(self, identification_number: int, *, x: float, y: float, node_type: NodeType = NodeType.HIDDEN):
        self.identification_number = identification_number
        self.node_type: NodeType = node_type
        self.x, self.y = x, y

    def __repr__(self):
        return f"NodeGene<x: {self.x} y: {self.y}>"
