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

from NEAT.Network import Connection


class NodeType(Enum):
    INPUT = 1
    HIDDEN = 2
    OUTPUT = 3


class Node:
    def __init__(self, x: float, y: float, *, node_type: NodeType, identification_number: int):
        self.identification_number = identification_number
        self.node_type: NodeType = node_type
        self.value = 0
        self.connections = []
        self.x = x
        self.y = y

    def add_connection(self, connection: Connection):
        self.connections.append(connection)

    def __repr__(self):
        return f"Node<x: {self.x} value: {self.value}>"
