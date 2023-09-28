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
from typing import Optional

from NEAT.GeneConnection import GeneConnection
from NEAT.GeneNode import GeneNode, GeneNodeType
from NEAT.Network import NeuralNetwork, Node, Connection


class NEAT:
    def __init__(self, structure: tuple[int, int], *, number_of_networks: int = 5):
        self.node_genes: dict[int, GeneNode] = {}
        self.connection_genes: dict[int, GeneConnection] = {}
        self.starting_structure = structure
        self.networks: list[NeuralNetwork] = []
        self.generation: int = 1
        self.__number_of_networks = number_of_networks
        self.__number_of_x: list[float] = [0, 1]

        self.__init_neat__(self.starting_structure)
        self.__init_networks__(self.starting_structure)

    def __init_neat__(self, structure: tuple[int, int]):
        inn_number = 0
        nodes = {}

        for i in range(structure[0]):
            nodes[inn_number] = GeneNode(
                identification_number=inn_number,
                node_type=GeneNodeType.INPUT,
                x=0, y=i / 10
            )
            inn_number += 1

        for i in range(structure[1]):
            nodes[inn_number] = GeneNode(
                identification_number=inn_number,
                node_type=GeneNodeType.OUTPUT,
                x=1, y=i / 10
            )
            inn_number += 1

        self.node_genes = nodes

    def __init_networks__(self, structure: tuple[int, int]):
        for i in range(self.__number_of_networks):
            self.networks.append(NeuralNetwork(structure, neat=self))

    def get_node(self, identification_number: int, x: float) -> GeneNode:
        if node := self.node_genes.get(identification_number):
            return node

        if x not in self.__number_of_x:
            self.__number_of_x.append(x)
            self.__number_of_x.sort()

        nodes_x = self.get_nodes_from_x(x)
        new_node_identification = self.get_new_node_identification()
        node = GeneNode(
            identification_number=new_node_identification,
            x=x, y=len(nodes_x) / 10
        )
        self.node_genes[new_node_identification] = node
        return node

    def get_node_from_xy(self, x: float, y: float) -> Optional[GeneNode]:
        nodes_x = self.get_nodes_from_x(x)
        for n in nodes_x:
            if n.y == y:
                return n
        return None

    def get_connection(self, identification_number: int) -> Optional[GeneConnection]:
        if connection := self.connection_genes.get(identification_number):
            return connection
        return None

    def get_connection_from_nodes(self, from_node: Node, to_node: Node) -> GeneConnection:
        for connection in self.connection_genes.values():
            if (connection.from_node.identification_number == from_node.identification_number and
                    connection.to_node.identification_number == to_node.identification_number):
                return connection

        new_identification_number = self.get_new_connection_identification()
        new_connection = GeneConnection(
            identification_number=new_identification_number,
            from_node=self.get_node(from_node.identification_number, from_node.x),
            to_node=self.get_node(to_node.identification_number, to_node.x)
        )
        self.connection_genes[new_identification_number] = new_connection
        return new_connection

    def remove_connection(self, identification_number: int):
        del self.connection_genes[identification_number]

    def get_new_connection_identification(self) -> int:
        return max(self.connection_genes.keys(), default=0) + 1

    def get_new_node_identification(self) -> int:
        return max(self.node_genes.keys(), default=0) + 1

    def get_nodes_from_x(self, x: float) -> list[GeneNode]:
        return list(filter(lambda n: n.x == x, self.node_genes.values()))

    def get_layer_amount(self) -> int:
        return len(self.__number_of_x)

    def get_layer_number(self, x: float) -> int:
        return self.__number_of_x.index(x)

    def next_generation(self):
        """
        Finish the current generation and select and mutate the networks.
        """
