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
import random

from NEAT.Network.Node import Node


class Connection:
    def __init__(self, from_node: Node, to_node: Node, *, identification_number: int):
        self.from_node: Node = from_node
        self.to_node: Node = to_node
        self.enabled: bool = True
        self.identification_number = identification_number
        self.weight: float = random.uniform(-2, 2)

    def __repr__(self):
        return f"Connection<from_x: {self.from_node.x} from_y: {self.from_node.y} to_x: {self.to_node.x} to_y: {self.to_node.y}>"
