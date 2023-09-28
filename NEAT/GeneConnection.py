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

from NEAT.GeneNode import GeneNode


class GeneConnection:
    def __init__(self, identification_number: int, *, from_node: GeneNode, to_node: GeneNode):
        self.identification_number: int = identification_number
        self.from_node: GeneNode = from_node
        self.to_node: GeneNode = to_node

    def __repr__(self):
        return f"ConnectionGene<from_x: {self.from_node.x} to_x: {self.to_node.x}"
