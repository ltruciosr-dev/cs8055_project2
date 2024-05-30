import time
import random
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
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

def weight(item: int):
    return paquetes[item].peso

def list_to_array(list: List[List[int]]):
    max_length = max(len(sublist) for sublist in list)
    padded_list = [sublist + [sublist[-1]] * (max_length - len(sublist)) for sublist in list]

    return np.array(padded_list)

def plot_fitness(list_iterations: List[List[int]], list_values: List[List[int]], list_pesos: List[List[int]]):
    fig = go.Figure()

    # Plot the fitness by container
    palette = px.colors.qualitative.Pastel
    idx = 0
    for sublist, subvalue, subpeso in zip(list_iterations, list_values, list_pesos):
        subfitness = np.array(subvalue) / np.array(subpeso)
        color = palette[idx % len(palette)]
        fig.add_trace(go.Scatter(x=sublist, y=subfitness, mode='lines+markers', name=f'Fitness Container {idx}', line=dict(color=color)))
        idx+=1

    # Plot the average fitness
    arr_iterations = np.arange(max(len(sublist) for sublist in list_iterations))
    arr_values = list_to_array(list_values).sum(axis=0)
    arr_pesos = list_to_array(list_pesos).sum(axis=0)
    arr_fitness = arr_values / arr_pesos

    fig.add_trace(go.Scatter(x=arr_iterations, y=arr_fitness, mode='lines+markers', 
                             name='Total Fitness', 
                             line=dict(width=5, color='black'),  # Customize line width and color
                             marker=dict(size=8)))
    
    # Add title and labels
    fig.update_layout(title='Simulated Annealing',
                    xaxis_title='Iterations',
                    yaxis_title='Fitness Values')
    
    # Show plot
    fig.show()

def get_random_neighbor(items: List[int], available_items: List[int], capacidad: int):
    residual_items = list(set(available_items) - set(items))
    neighbor = items.copy()
    random_item = random.choice(neighbor)
    neighbor.remove(random_item)
    while peso_total(neighbor) <= capacidad:
        if not residual_items:
            break
        residual_item = random.choice(residual_items)
        residual_items.remove(residual_item)
        neighbor.append(residual_item)
    while peso_total(neighbor) > capacidad:
        remove_item = random.choice(neighbor)
        neighbor.remove(remove_item)
    return neighbor

def accept(delta: float, T: float):
  if delta > 0:
    return True
  else:
    r = np.random.rand()
    if r < np.exp(delta/T):
      return True
    else:
      return False


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

    # Parameters
    list_iterations = []
    list_values = []
    list_pesos = []
    for idx, container in enumerate(containers):
        print(f"\nContainer {idx}")
        available_items = residual_items.copy()
        remove_items = list(set(container.items)-set(available_items))
        for r_item in remove_items: container.items.remove(r_item)
        if not container.items: container.items = [random.choice(available_items)]
        print(f"- capacidad: {container.capacidad}")
        print(f"- initial items: {container.items}")
        
        # Start Iteration
        iteration = 0
        T_max = 1000
        T_min = 0.1
        cooling_rate = 0.90
        E = fitness(container.items)

        iterations = [iteration]
        values = [fitness_valor(container.items)]
        pesos = [fitness_peso(container.items)]
        while T_max > T_min:
            neighbor = get_random_neighbor(
                items = container.items,
                available_items=available_items, 
                capacidad=container.capacidad)
            E_new = fitness(neighbor)
            delta = E_new - E
            print(f"Iteration {iteration}")
            print(f"- items: {container.items}, score: {fitness(container.items)}, weight: {fitness_peso(container.items)}, valor: {fitness_valor(container.items)}")
            print(f"- neighbor: {neighbor}, score: {fitness(neighbor)}, weight: {fitness_peso(neighbor)}, valor: {fitness_valor(neighbor)}")
            if accept(delta, T_max):
                print(f"- neighbor accepted, T_max:{T_max}")
                container.items = neighbor
                E = E_new
            T_max *= cooling_rate
            iteration += 1
            valor = fitness_valor(container.items)
            peso = fitness_peso(container.items)
            
            # Store in logs
            iterations.append(iteration)
            values.append(valor)
            pesos.append(peso)

        list_iterations.append(iterations)
        list_values.append(values)
        list_pesos.append(pesos)

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
    simulated_annealing_time = end_time - start_time
    print(f"\nSimulated Annealing elapsed time: {simulated_annealing_time} seconds")

    # Plot fitness evolution
    plot_fitness(list_iterations, list_values, list_pesos)
    