# Frozen Lake Solver From First Principles Using Q-Learning

## Introduction
* **What is Reinforcement Learning?** 
  Reinforcement Learning (RL) is a branch of machine learning where an AI agent learns by trial and error. The agent interacts with an environment by making moves, receiving rewards for good actions, and penalties for bad ones. Over time, it learns the best sequence of choices to maximize its long-term total rewards.

* **What is Frozen Lake?** 
  Frozen Lake is a grid-world puzzle game where a player must walk across an 8×8 grid. The agent starts at a designated safe space (`S`) and must find a path to a safe cabin goal tile (`G`). Along the way, the agent must avoid stepping on lethal water holes (`H`) that end the game immediately, while stepping on normal frozen ice tiles (`F`) is completely safe.

---

## Environment Design
* **State Representation**: 
  The environment consists of 64 distinct squares on an 8×8 grid. Each square is represented by a single integer state value ranging from 0 (the top-left corner) to 63 (the bottom-right goal corner). This unique index is calculated using a clean mathematical mapping: `State = (Row * 8) + Column`.

* **Action Representation**: 
  The agent can choose from 4 discrete, directional moves at any step:
  * `0`: Move Left
  * `1`: Move Down
  * `2`: Move Right
  * `3`: Move Up

* **Reward Structure**: 
  The environment awards values based solely on the final landing cell destination:
  * Reaching the Goal tile (`G`): `+1.0` reward.
  * Stepping into a Hole tile (`H`): `0.0` reward.
  * Standing on a Frozen tile (`F`): `0.0` reward.

---

## Q-Learning Algorithm
* **Description of Q-Learning**: 
  Q-Learning is an off-policy value-based reinforcement learning algorithm. It tracks the "quality" or expected value of taking a specific action within a given state. These values are stored inside a 2D memory array called a Q-table, where rows represent the 64 states and columns represent the 4 actions.

* **Explanation of the Update Equation**: 
  The table updates its value matrices step-by-step using the exact mathematical equation:
  `Q(s,a) = Q(s,a) + alpha * [r + (gamma * max Q(s', a')) - Q(s,a)]`
  * `alpha (Learning Rate)`: Controls how much weight the agent places on brand new experiences over past estimates.
  * `gamma (Discount Factor)`: Determines how much the agent cares about long-term future targets versus immediate gains.
  * `max Q(s', a')`: Looks ahead to the next state and predicts the highest possible reward it can achieve from there.

* **Exploration Strategy**: 
  This model utilizes an Epsilon-Greedy system to balance exploration (searching for new paths) and exploitation (using known safe routes). A tracking variable called `epsilon` sets the probability of making a random move. When a randomly generated number falls below epsilon, the agent explores; otherwise, it exploits the highest available number in its table.

---

## Training Procedure
* **Hyperparameters Used**:
  * **Learning Rate (\(\alpha\))**: 0.1
  * **Discount Factor (\(\gamma\))**: 0.95
  * **Epsilon Decay Rate**: 0.995 (reduces exploration slowly over time)
  * **Minimum Epsilon Floor**: 0.01 (ensures a tiny amount of exploration remains)

* **Number of Episodes**: 
  Both testing agents were run through exactly 15,000 training episodes to give them enough time to safely find the goal.

---

## Results
* **Final Success Rate**: 
  The agent scored a perfect **100.00% success rate** over 100 test evaluation runs. This allowed for efficient and safe navigation from start to finish with no failures.

* **Learned Policy**: 
  The visual arrow map below is extracted from the trained agent's Q-table, which shows the safe choices it learned for the grid layout:
  ```text

  |  →  |  →  |  →  |  ↓  |  ↓  |  ←  |  ?  |  ?  |
  |  ↑  |  ←  |  ↑  |  →  |  →  |  ↓  |  ↓  |  ↓  |
  |  ?  |  ?  |  ?  |  H  |  →  |  ↓  |  ←  |  ←  |
  |  ?  |  ?  |  ?  |  H  |  ↑  |  →  |  ↓  |  ↑  |
  |  ?  |  ?  |  ?  |  H  |  ?  |  →  |  →  |  ↓  |
  |  ?  |  H  |  H  |  ?  |  ?  |  ?  |  H  |  ↓  |
  |  ?  |  H  |  ?  |  ?  |  H  |  ?  |  H  |  ↓  |
  |  ?  |  ?  |  ?  |  H  |  ?  |  ?  |  ?  |  G  |
  ```
  *(Note: `?` symbols denote unvisited states. The agent found a clean, safe path early on, so it never needed to waste time exploring those specific squares).*

* **Discussion of Performance**: 
  The **Decaying Epsilon** agent performed significantly better during training, achieving a stable 99% success rate by episode 2,000. Because its exploration rate shrank over time, it quickly locked onto efficient, repeatable paths. In contrast, the **Pure Epsilon** agent hovered around an 80% success rate during training because it was forced to make random exploratory mistakes 20% of the time. However, once exploration was disabled for the final evaluation phase, both models achieved 100% accuracy, showing that both successfully learned the core solution.

---

## Execution Instructions
The below terminal commands are folowed to run the project scripts in an orderly and sequential manner:

1. **Train the agent models and export performance data**:
   ```bash
   python train.py
   ```
2. **Evaluate the saved policy maps and print the visual map arrays**:
   ```bash
   python evaluate.py
   ```
