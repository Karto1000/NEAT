import pygame

from NEAT import NEAT

pygame.init()
SW, SH = 1200, 1000
SCREEN = pygame.display.set_mode((SW, SH))
NN_W, NN_H = 500, 200
FONT = pygame.font.SysFont("Arial", 20)

current_network = 0
number_of_networks = 10

neat = NEAT((3, 3), number_of_networks=number_of_networks)
neat.networks[0].add_random_connection()

while True:
    SCREEN.fill((255, 255, 255))

    for e in pygame.event.get():
        if e.type == pygame.QUIT:
            pygame.quit()
            exit(0)
        if e.type == pygame.KEYDOWN:
            if e.key == pygame.K_RIGHT:
                if current_network == number_of_networks - 1:
                    current_network = 0
                else:
                    current_network += 1
            elif e.key == pygame.K_LEFT:
                if current_network == 0:
                    current_network = number_of_networks - 1
                else:
                    current_network -= 1
            if e.key == pygame.K_SPACE:
                neat.networks[current_network].add_random_node()
            if e.key == pygame.K_c:
                neat.networks[current_network].add_random_connection()
            if e.key == pygame.K_p:
                print(neat.networks[current_network].propagate([1, 2, 3]))

    layer_amount = neat.get_layer_amount()

    # for connection in neat.connection_genes.values():
    #     from_layer = neat.get_layer_number(connection.from_node.x)
    #     to_layer = neat.get_layer_number(connection.to_node.x)
    #
    #     pygame.draw.line(
    #         SCREEN,
    #         (0, 0, 0),
    #         (NN_W * (1 / layer_amount * from_layer) + 50, connection.from_node.y * 250 + NN_H),
    #         (NN_W * (1 / layer_amount * to_layer) + 50, connection.to_node.y * 250 + NN_H),
    #     )
    #
    # current_layer = 0
    # last_x = 0
    # for node in sorted(neat.node_genes.values(), key=lambda e: e.x):
    #     if node.x > last_x:
    #         current_layer += 1
    #         last_x = node.x
    #
    #     pygame.draw.circle(
    #         SCREEN,
    #         (0, 0, 0),
    #         (NN_W * (1 / layer_amount * current_layer) + 50, node.y * 250 + NN_H),
    #         10
    #     )

    network = neat.networks[current_network]

    for connection in network.connections:
        from_layer = network.get_layer_number(connection.from_node.x)
        to_layer = network.get_layer_number(connection.to_node.x)

        pygame.draw.line(
            SCREEN,
            (0, 0, 0) if connection.weight > 0 else (255, 0, 0),
            (NN_W * (1 / layer_amount * from_layer) + 50, connection.from_node.y * 250 + 50),
            (NN_W * (1 / layer_amount * to_layer) + 50, connection.to_node.y * 250 + 50),
            max(1, int(abs(connection.weight * 2)))
        )

    current_layer = 0
    last_x = 0
    for node in network.nodes:
        if node.x > last_x:
            current_layer += 1
            last_x = node.x

        pygame.draw.circle(
            SCREEN,
            (0, 0, 0),
            (NN_W * (1 / layer_amount * current_layer) + 50, node.y * 250 + 50),
            10
        )

    text = FONT.render(f"Current network: {current_network + 1}", False, (0, 0, 0))
    SCREEN.blit(text, (45, 10))

    pygame.display.update()
