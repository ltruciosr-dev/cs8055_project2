# CS8055 Project 2: Comparison between Hill Climbing and Simulated Annealing

## Project Overview

This project aims to compare the performance of two optimization algorithms, Hill Climbing and Simulated Annealing, in the context of a container packing problem. The goal is to maximize the fitness of packages placed into containers. The fitness is defined as the ratio of the sum of package values to the sum of package weights.

## Problem Description

Given `N_PACKAGES`, each with a specific value and weight, and `N_CONTAINERS`, the task is to optimize the placement of packages into containers such that the fitness is maximized. We start with an initial solution and then apply the two heuristics, Hill Climbing and Simulated Annealing, to iteratively improve the solution.

### A. Hill Climbing Heuristic

Hill Climbing is an iterative algorithm that starts with an arbitrary solution to a problem and attempts to find a better solution by incrementally making small changes to the current solution. The process is repeated until no further improvements can be made.

#### Algorithm Steps

1. Start with an initial solution.
2. Generate neighbors by making small changes to the current solution.
3. Evaluate the fitness of each neighbor.
4. Move to the neighbor with the highest fitness.
5. Repeat steps 2-4 until no further improvements can be made.

```python
def get_neighbors(items: List[int], available_items: List[int], capacidad: int):
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
```

### B. Simulated Annealing Heuristic

Simulated Annealing is a probabilistic technique for approximating the global optimum of a given function. It allows for occasional acceptance of worse solutions, which helps to avoid being trapped in local optima.

#### Algorithm Steps

1. Start with an initial solution.
2. Generate a random neighbor of the current solution.
3. Evaluate the fitness of the neighbor.
4. Decide whether to move to the neighbor based on the acceptance probability, which decreases over time.
5. Repeat steps 2-4, gradually reducing the acceptance probability, until the stopping criterion is met.

```python
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
```

## Evaluation and Comparison

The performance of both Hill Climbing and Simulated Annealing was evaluated based on a test case involving 100 containers and 1000 packages. The key metrics for evaluation were the final fitness value achieved and the inference time required by each algorithm.

### Results
| Algorithm | Final Fitness | Avg Inference Time |
| --------- | ------------- | ------------------ |
| Hill Climbing	| 2.13 | **77 ms** |
| Simulated Annealing | **2.17** | 812 ms |

### Analysis

- Final Fitness: Simulated Annealing outperformed Hill Climbing by achieving a higher final fitness value of 2.17 compared to Hill Climbing's fitness value of 2.13. This indicates that Simulated Annealing is more effective in finding a better solution in terms of maximizing the fitness of packages within the containers.

- Inference Time: Hill Climbing demonstrated superior speed with an inference time of 0.08 seconds, significantly faster than Simulated Annealing's inference time of 0.8 seconds. This highlights Hill Climbing's efficiency in reaching a solution more quickly.

## Conclusion

In this comparison:

- Simulated Annealing is more effective in optimizing the fitness of the packages, making it the preferred choice when the quality of the solution is of utmost importance, despite its longer inference time.
- Hill Climbing is more efficient in terms of computation time, making it suitable for scenarios where quick results are required, even if the solution is slightly less optimal.

This evaluation provides a clear understanding of the trade-offs between the two algorithms, allowing for informed decision-making based on the specific requirements of the container packing problem.