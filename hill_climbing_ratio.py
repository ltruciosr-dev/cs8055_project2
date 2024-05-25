import random
import time
import numpy as np
from typing import List
from dataclasses import dataclass

# Define the data structure using dataclass
@dataclass
class Paquete:
    indice: int
    peso: int
    valor: int

@dataclass
class Container:
    indice: int
    capacidad: int
    valor: int = 0
    peso: int = 0
    items: List[int] = None

def peso_total(items: List[int]):
    return np.sum([paquetes[item].peso for item in items])

def fitness_peso(items: List[int]):
    return np.sum([paquetes[item].peso for item in items])

def fitness_valor(items: List[int]):
    return np.sum([paquetes[item].valor for item in items])

def fitness(items: List[int]):
    return fitness_valor(items) / fitness_peso(items)

def total_fitness_valor(containers: List[Container]):
    return np.sum([container.valor for container in containers])

def total_fitness_peso(containers: List[Container]):
    return np.sum([container.peso for container in containers])

def total_fitness(containers: List[Container]):
    return total_fitness_valor(containers) / total_fitness_peso(containers)

def get_neighbors(items: List[int], available_items: List[int], capacidad: int):
    """
    Generate a list of neighboring item configurations within the given capacity.

    Args:
        items (List[int]): The current list of item indices.
        available_items (List[int]): The list of available item indices.
        capacidad (int): The maximum capacity for the items.

    Returns:
        List[List[int]]: A list of neighboring item configurations.
    """
    residual_items = list(set(available_items) - set(items))
    neighbors = []
    for item in items:
        new_items = items.copy()
        new_items.remove(item)
        while peso_total(new_items) <= capacidad:
            if not residual_items:
                break
            residual_item = random.choice(residual_items)
            residual_items.remove(residual_item)
            new_items.append(residual_item)
        while peso_total(new_items) > capacidad:
            remove_item = random.choice(new_items)
            new_items.remove(remove_item)
        neighbors.append(new_items)
    return neighbors


if __name__ == '__main__':
    # Seed for reproducibility
    random.seed(0)
    np.random.seed(0)

    N_CONTAINERS = 100 # 3
    N_PAQUETES = 1000 # 20

    # Define container capacities
    containers_capacidad = np.random.randint(5, 10, N_CONTAINERS)
    containers = [Container(indice, capacidad) for indice, capacidad in zip(range(N_CONTAINERS), containers_capacidad)]
    containers.sort(key=lambda containers: containers.capacidad, reverse=True)

    for container in containers:
        print(container)

    # Define packages by weight and value
    paquetes_peso = np.random.randint(1, 4, N_PAQUETES)
    paquetes_valor = np.random.randint(2, 5, N_PAQUETES)
    
    paquetes = [Paquete(indice, peso, valor) for peso, valor, indice in zip(paquetes_peso, paquetes_valor, range(N_PAQUETES))]

    for paquete in paquetes:
        print(paquete)

    # Set a random initialization
    start_ix = 0 

    for container in containers:
        peso_acumulativo = np.cumsum([paquetes.peso for paquetes in paquetes[start_ix:]])
        cantidad_paquetes = len(peso_acumulativo[peso_acumulativo <= container.capacidad])
        container.items = [paquete.indice for paquete in paquetes[start_ix:start_ix + cantidad_paquetes]]
        container.valor = np.sum([paquete.valor for paquete in paquetes[start_ix:start_ix + cantidad_paquetes]])
        container.peso = np.sum([paquete.peso for paquete in paquetes[start_ix:start_ix + cantidad_paquetes]])
        start_ix += cantidad_paquetes

    for container in containers:
        print(container)

    # Start simulation
    start_time = time.time()

    fitness_inicial = total_fitness(containers)
    print(f"\nInitial Fitness: {total_fitness(containers)}")

    residual_items = [paquete.indice for paquete in paquetes]

    for idx, container in enumerate(containers):
        print(f"\nContainer {idx}")
        available_items = residual_items.copy()
        remove_items = list(set(container.items)-set(available_items))
        for r_item in remove_items: container.items.remove(r_item)
        print(f"- capacidad: {container.capacidad}")
        print(f"- initial items: {container.items}")
        
        # Start Iteration
        iteration = 0
        while True:
            neighbors = get_neighbors(
                items = container.items, 
                available_items=available_items, 
                capacidad=container.capacidad)
            if not neighbors:
                break
            best_neighbor = max(neighbors, key = fitness)
            print(f"Iteration {iteration}")
            print(f"- items: {container.items}, score: {fitness(container.items)}, weight: {fitness_peso(container.items)}, valor: {fitness_valor(container.items)}")
            print(f"- neighbors: {neighbors}")
            print(f"- best neighbor: {best_neighbor}, score: {fitness(best_neighbor)}, weight: {fitness_peso(best_neighbor)}, valor: {fitness_valor(container.items)}")
            if fitness(best_neighbor) >= fitness(container.items):
                remove_items = list(set(container.items)-set(best_neighbor))
                container.items = best_neighbor
                for r_item in remove_items: available_items.remove(r_item)
            else:
                break
            iteration += 1
        residual_items = list(set(residual_items) - set(container.items))
        container.valor = fitness_valor(container.items)
        container.peso = fitness_peso(container.items)
        print(f"Best Items: {container.items}")

    # Print the contents of each container
    for container in containers:
        print(container)

    print(f"\nInitial Fitness: {fitness_inicial}")
    print(f"Final Fitness: {total_fitness(containers)}")

    end_time = time.time()
    hill_climbing_time = end_time - start_time
    print(f"\nHill Climbing elapsed time: {hill_climbing_time} seconds")
    