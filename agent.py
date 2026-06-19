# agent.py
import random

class QLearningAgent:
    def __init__(self, num_states, num_actions, alpha=0.1, gamma=0.99, 
                 epsilon_start=1.0, epsilon_min=0.01, epsilon_decay=0.995, strategy="decaying"):
        self.num_states = num_states
        self.num_actions = num_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.strategy = strategy  

        # Initialize Q-Table with zeros 
        self.q_table = [[0.0 for _ in range(num_actions)] for _ in range(num_states)]

    def choose_action(self, state):
        """Selects action using Epsilon-Greedy policy exploration."""
        if random.random() < self.epsilon:
            return random.randint(0, self.num_actions - 1) 
        else:
            return self._get_max_action(state) 

    def _get_max_action(self, state):
        """Helper to get action with highest Q-value; breaks ties randomly."""
        max_val = max(self.q_table[state])
        actions_with_max = [a for a, q in enumerate(self.q_table[state]) if q == max_val]
        return random.choice(actions_with_max)

    def update(self, state, action, reward, next_state, done):
        """Updates the Q-Table using Bellman optimality updates."""
        current_q = self.q_table[state][action]
        max_next_q = max(self.q_table[next_state]) if not done else 0.0
        
        # Exact mathematical equation:
        # Q(s,a) <- Q(s,a) + alpha * [r + gamma * max(Q(s', a')) - Q(s,a)]
        self.q_table[state][action] = current_q + self.alpha * (reward + (self.gamma * max_next_q) - current_q)

    def decay_epsilon(self):
        """Handles explicit step decay sequences."""
        if self.strategy == "decaying":
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        # If strategy is "pure", epsilon stays fixed at epsilon_start
