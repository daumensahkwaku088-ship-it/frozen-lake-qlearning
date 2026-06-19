# environment.py

class FrozenLakeEnv:
    def __init__(self):
        
        self.map = [
            "SFFFFFFF",
            "FFFFFFFF",
            "FFFHFFFF",
            "FFFHFFFF",
            "FFFHFFFF",
            "FHHFFFHF",
            "FHFFHFHF",
            "FFFHFFFG"
        ]
        self.rows = 8
        self.cols = 8
        self.num_states = self.rows * self.cols
        self.num_actions = 4  
        
        self.action_effects = {
            0: (0, -1),  
            1: (1, 0),   
            2: (0, 1),   
            3: (-1, 0)   
        }
        
        self.reset()

    def reset(self):
        """Resets the environment to the starting position (0,0)."""
        self.current_row = 0
        self.current_col = 0
        return self.get_state()

    def get_state(self):
        """Returns the current state as a single flattened integer index."""
        return self.current_row * self.cols + self.current_col

    def _get_coords_from_state(self, state):
        """Helper to convert integer state back to grid coordinates."""
        return divmod(state, self.cols)

    def is_terminal(self):
        """Checks if current state is a Hole (H) or Goal (G)."""
        current_cell = self.map[self.current_row][self.current_col]
        return current_cell in ('H', 'G')

    def step(self, action):
        """Advances the environment state by executing an action."""
        if self.is_terminal():
            return self.get_state(), 0.0, True

        dr, dc = self.action_effects[action]
        new_row = self.current_row + dr
        new_col = self.current_col + dc

        if 0 <= new_row < self.rows and 0 <= new_col < self.cols:
            self.current_row = new_row
            self.current_col = new_col

        current_cell = self.map[self.current_row][self.current_col]
        done = self.is_terminal()
        
        # Reward Structure
        if current_cell == 'G':
            reward = 1.0
        elif current_cell == 'H':
            reward = 0.0
        else:
            reward = 0.0

        return self.get_state(), reward, done

    def render(self):
        """Renders the grid environment textually in the console."""
        for r in range(self.rows):
            row_str = ""
            for c in range(self.cols):
                if r == self.current_row and c == self.current_col:
                    row_str += "🤖 "  
                else:
                    row_str += f"{self.map[r][c]} "
            print(row_str)
        print("-" * 20)
