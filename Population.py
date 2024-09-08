import os
import torch
import torch.nn as nn
import random


class Creature(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=64):
        super(Creature, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
        )
        self.fitness = 0

    def forward(self, x):
        return self.network(x)

    def mutate(self, mutation_rate=0.1, mutation_strength=0.1):
        with torch.no_grad():
            for param in self.parameters():
                if random.random() < mutation_rate:
                    noise = torch.randn_like(param) * mutation_strength
                    param.add_(noise)

    def clone(self):
        new_one = Creature(
            self.network[0].in_features,
            self.network[-1].out_features,
            self.network[0].out_features,
        )
        new_one.load_state_dict(self.state_dict())
        return new_one

    def set_fitness(self, fitness):
        self.fitness = fitness

    def increase_fitness(self, fitness):
        self.fitness += fitness

    def act(self, state):
        with torch.no_grad():
            self.eval()
            logits = self.forward(state)
            return torch.argmax(logits).item()


class Population:
    def __init__(self, size, creature_args):
        self.creature_args = creature_args
        self.creatures = [Creature(**creature_args) for _ in range(size)]
        self.best_creatures = []
        self.best_fitness = []
        self.mutate_rate = 0.1
        self.mutate_decay = 0.999

    def get_creatures(self):
        return self.creatures

    def act(self, states):
        return [creature.act(state) for creature, state in zip(self.creatures, states)]

    def reset(self):
        return

    def evolve(self, elite_size=0.05):
        # Fitness evaluation
        fitness_scores = [
            self.evaluate_fitness(creature) for creature in self.creatures
        ]

        total_fitness = sum(fitness_scores)
        if total_fitness == 0:
            probabilities = [1 / len(fitness_scores)] * len(
                fitness_scores
            )  # All equal chance if total fitness is 0
        else:
            probabilities = [fitness / total_fitness for fitness in fitness_scores]

        self.best_fitness.append(max(fitness_scores))
        print(f"Best fitness: {self.best_fitness[-1]}")

        # Track the best creature
        self.best_creatures.append(
            self.creatures[fitness_scores.index(max(fitness_scores))].clone()
        )

        # Select the elite survivors (without breeding)
        sorted_indices = sorted(
            range(len(self.creatures)), key=lambda i: fitness_scores[i], reverse=True
        )
        survivors = [
            self.creatures[i].clone()
            for i in sorted_indices[: int(len(self.creatures) * elite_size)]
        ]

        self.mutate_rate *= self.mutate_decay  # Decay mutation rate over time

        # Breeding and mutation to refill the population
        while len(survivors) < len(self.creatures):
            parent1 = self.roulette_wheel_selection(probabilities)
            parent2 = self.roulette_wheel_selection(probabilities)
            child = self.breed(parent1, parent2)
            child.mutate(mutation_rate=self.mutate_rate)
            child.set_fitness(0)
            survivors.append(child)

        self.creatures = survivors

    def roulette_wheel_selection(self, probabilities):
        """Select a creature based on fitness proportionate selection."""
        selection = random.choices(self.creatures, weights=probabilities, k=1)
        return selection[0]

    def breed(self, parent1, parent2):
        child = parent1.clone()
        for child_param, parent2_param in zip(child.parameters(), parent2.parameters()):
            if random.random() > 0.5:
                child_param.data.copy_(parent2_param.data)
        return child

    def evaluate_fitness(self, creature):
        # Placeholder: Implement the actual interaction with the environment
        return creature.fitness  # Random fitness for placeholder

    def save(self, filename):
        torch.save(
            {
                "creatures": [creature.state_dict() for creature in self.creatures],
                "highest_fitnesses": self.best_fitness,
                "best_creatures": [
                    creature.state_dict() for creature in self.best_creatures
                ],
                "mutation_rate": self.mutate_rate,
                "mutation_decay": self.mutate_decay,
            },
            filename,
        )

    def load(self, filename):
        if not os.path.exists(filename):
            return
        checkpoint = torch.load(filename)
        self.creatures = [
            Creature(**self.creature_args) for _ in checkpoint["creatures"]
        ]
        self.best_fitness = checkpoint["highest_fitnesses"]
        self.best_creatures = [
            Creature(**self.creature_args) for _ in checkpoint["best_creatures"]
        ]
        for creature, state_dict in zip(
            self.best_creatures, checkpoint["best_creatures"]
        ):
            creature.load_state_dict(state_dict)
        for creature, state_dict in zip(self.creatures, checkpoint["creatures"]):
            creature.load_state_dict(state_dict)
        self.mutate_rate = checkpoint["mutation_rate"]
        self.mutate_decay = checkpoint["mutation_decay"]
