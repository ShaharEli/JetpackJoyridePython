# Jetpack Joyride Remake in Python

This project is a Python-based remake of the popular game Jetpack Joyride. It includes various scripts and data files to simulate and analyze the performance of different strategies in the game.

## Project Structure

### Files and Directories

- `assets/`: Directory containing game assets.
  - `sprites/`: Directory containing sprite images.
    - `skins/`: Directory containing skin images for sprites.
- `evolution_agent_without_rockets_scores.json`: JSON file containing scores for evolution agents without rockets.
- `evolution_with_rockets_scores.json`: JSON file containing scores for evolution agents with rockets.
- `main.py`: Main script to run the game.
- `player_info.txt`: Text file containing player information.
- `population.pth`: File containing saved population data.
- `Population.py`: Script defining the `Population` class used in the game.
- `random_scores_with_rockets.json`: JSON file containing random scores with rockets.
- `random_scores_without_rockets.json`: JSON file containing random scores without rockets.
- `requirements.txt`: File listing the Python dependencies for the project.
- `research.ipynb`: Jupyter Notebook for analyzing and visualizing the performance of different strategies.

### How to Run

1. Install the required dependencies:
    ```sh
    pip install -r requirements.txt
    ```

2. Run the game:
    ```sh
    python main.py
    ```

3. Analyze the results using the Jupyter Notebook:
    ```sh
    jupyter notebook research.ipynb
    ```

### Main Classes and Functions

- `Game` class in [`main.py`](main.py): Manages the game logic and state.
- `Population` class in [`Population.py`](Population.py): Manages the population of agents in the game.

### Data Files

- `evolution_agent_without_rockets_scores.json`: Contains scores for agents evolved without rockets.
- `evolution_with_rockets_scores.json`: Contains scores for agents evolved with rockets.
- `random_scores_with_rockets.json`: Contains random scores with rockets.
- `random_scores_without_rockets.json`: Contains random scores without rockets.

### Analysis

The `research.ipynb` notebook contains code to load and analyze the scores from the JSON files, and visualize the performance of different strategies using matplotlib.
