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
        self.best_creature = None
        self.best_fitness = 0

    def get_creatures(self):
        return self.creatures

    def act(self, states):
        return [creature.act(state) for creature, state in zip(self.creatures, states)]

    def reset(self):
        return

    def evolve(self, elite_size=0.2, mutation_rate=0.1):
        # Fitness evaluation
        fitness_scores = [
            self.evaluate_fitness(creature) for creature in self.creatures
        ]
        sorted_indices = sorted(
            range(len(self.creatures)), key=lambda i: fitness_scores[i], reverse=True
        )
        survivors = [
            self.creatures[i].clone()
            for i in sorted_indices[: int(len(self.creatures) * elite_size)]
        ]

        # Breeding and mutation to refill the population
        while len(survivors) < len(self.creatures):
            parent1, parent2 = random.sample(self.creatures, 2)
            child = self.breed(parent1, parent2)
            child.mutate(mutation_rate)
            child.set_fitness(0)
            survivors.append(child)

        self.creatures = survivors

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
            },
            filename,
        )

    def load(self, filename):
        checkpoint = torch.load(filename)
        self.creatures = [Creature(self.creature_args) for _ in checkpoint["creatures"]]
        for creature, state_dict in zip(self.creatures, checkpoint["creatures"]):
            creature.load_state_dict(state_dict)
