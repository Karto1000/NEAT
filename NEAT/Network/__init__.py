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
import math
import random
from typing import Optional

from NEAT.Network.Connection import Connection
from NEAT.Network.Node import Node, NodeType


def sigmoid(x: float) -> float:
    return 1 / (1 + 1 / math.exp(x))


class NeuralNetwork:
    def __init__(self, structure: tuple[int, int], neat):
        self.nodes = []
        self.connections = []
        self.neat = neat
        self.structure = structure
        self.__init_network__(structure)
        self.__number_of_x: list[float] = [0, 1]
        self.fitness = 0

    def __init_network__(self, structure: tuple[int, int]):
        inn_number = 0
        input_nodes = []
        output_nodes = []

        for i in range(structure[0]):
            input_nodes.append(Node(0, i / 10, identification_number=inn_number, node_type=NodeType.INPUT))
            inn_number += 1

        for i in range(structure[1]):
            output_nodes.append(Node(1, i / 10, identification_number=inn_number, node_type=NodeType.OUTPUT))
            inn_number += 1

        self.nodes.extend(input_nodes)
        self.nodes.extend(output_nodes)

    def add_connection_between(self, from_node: Node, to_node: Node) -> Connection:
        gene_connection = self.neat.get_connection_from_nodes(from_node, to_node)

        connection = Connection(
            from_node=from_node,
            to_node=to_node,
            identification_number=gene_connection.identification_number
        )

        from_node.add_connection(connection)

        self.connections.append(connection)

        return connection

    def add_node_between(self, connection: Connection, x: float) -> tuple[Node, Connection, Connection]:
        amount_of_nodes_in_x = len(self.get_nodes_from_x(x))
        existing_node = self.neat.get_node_from_xy(x, amount_of_nodes_in_x / 10)

        id_num = existing_node.identification_number if existing_node else self.neat.get_new_node_identification()

        new_node = Node(
            identification_number=id_num,
            x=x,
            y=amount_of_nodes_in_x / 10,
            node_type=NodeType.HIDDEN
        )

        gene_connection_before = self.neat.get_connection_from_nodes(
            from_node=connection.from_node,
            to_node=new_node
        )
        connection_before = Connection(
            from_node=connection.from_node,
            to_node=new_node,
            identification_number=gene_connection_before.identification_number
        )
        connection.from_node.add_connection(connection_before)
        connection_before.weight = 0

        gene_connection_after = self.neat.get_connection_from_nodes(
            from_node=new_node,
            to_node=connection.to_node
        )
        connection_after = Connection(
            from_node=new_node,
            to_node=connection.to_node,
            identification_number=gene_connection_after.identification_number
        )
        new_node.add_connection(connection_after)
        connection_after.weight = connection.weight

        self.connections.remove(connection)
        connection.from_node.connections.remove(connection)

        self.connections.append(connection_before)
        self.connections.append(connection_after)
        self.nodes.append(new_node)

        if new_node.x not in self.__number_of_x:
            self.__number_of_x.append(new_node.x)
            self.__number_of_x.sort()

        self.nodes.sort(key=lambda e: e.x)

        return new_node, connection_before, connection_after

    def toggle_random_connection(self) -> Connection:
        """
        Toggle a random connections enabled flag

        :return: The selected connection with the updated enabled flag
        """

        random_connection = random.choice(self.connections)
        self.toggle_connection(random_connection)
        return random_connection

    def shift_random_weight(self) -> Connection:
        """
        Shift a random connections weight

        :return: The selected connection
        """

        random_connection = random.choice(self.connections)
        self.shift_weight(random_connection)
        return random_connection

    def replace_random_weight(self) -> Connection:
        """
        Replace the weight of a random connection

        :return: The selected connection
        """

        random_connection = random.choice(self.connections)
        self.replace_weight(random_connection)
        return random_connection

    def add_random_node(self) -> tuple[Node, Connection, Connection]:
        """
        Add a random node between two connections

        :return: A tuple of the newly created node and the two new connections
        """

        if len(self.connections) == 0:
            print("No connections, adding random connection")
            self.add_random_connection()

        connection = random.choice(self.connections)
        x = (connection.from_node.x + connection.to_node.x) / 2
        return self.add_node_between(connection, x)

    def add_random_connection(self) -> Optional[Connection]:
        """
        Add a connection between two random nodes

        :return: The created connection
        """

        first_node = random.choice(list(filter(lambda n: n.x < 1, self.nodes)))
        nodes_with_greater_x = list(
            filter(lambda n: n.x > first_node.x
                             and n not in map(lambda c: c.to_node, first_node.connections), self.nodes))

        if len(nodes_with_greater_x) == 0:
            print("No nodes to connect")
            return None

        second_node = random.choice(nodes_with_greater_x)
        return self.add_connection_between(first_node, second_node)

    def get_amount_of_layers(self) -> int:
        return len(self.__number_of_x)

    def get_nodes_from_x(self, x: float) -> list[Node]:
        return list(filter(lambda n: n.x == x, self.nodes))

    def get_layer_number(self, x: float) -> int:
        return self.__number_of_x.index(x)

    def add_fitness(self, fitness: int):
        self.fitness += fitness

    def propagate(self, inputs: list[float]) -> list[float]:
        if len(inputs) != self.structure[0]:
            raise Exception(f"Expected {self.structure[0]} inputs but got {len(inputs)}")

        # Set the values for the inputs
        for i in range(0, self.structure[0]):
            self.nodes[i].value = inputs[i]

        for n in self.nodes:
            if n.node_type != NodeType.INPUT:
                n.value = sigmoid(n.value)

            for c in n.connections:
                c.to_node.value += n.value * c.weight

        outputs = []
        for i in range(len(self.nodes) - 1, len(self.nodes) - 1 - self.structure[1], -1):
            outputs.append(self.nodes[i].value)

        self.reset_node_values()

        return outputs

    def reset_node_values(self):
        for n in self.nodes:
            n.value = 0

    @staticmethod
    def toggle_connection(connection: Connection) -> bool:
        """
        Toggle a random connections enabled flag

        :param connection: The connection to toggle
        :return: The new enabled status
        """

        connection.enabled = not connection.enabled
        return connection.enabled

    @staticmethod
    def shift_weight(connection: Connection) -> float:
        """
        Shift the weight of a connection

        :param connection: The connection to shift the weight of
        :return: The new weight of the connection
        """

        connection.weight += min(2, max(-2, random.uniform(-0.5, 0.5)))
        return connection.weight

    @staticmethod
    def replace_weight(connection: Connection) -> float:
        """
        Replace the weight of a connection with a new weight

        :param connection: The connection to replace the weight of
        :return: The new weight of the connection
        """

        connection.weight = random.uniform(-2, 2)
        return connection.weight
