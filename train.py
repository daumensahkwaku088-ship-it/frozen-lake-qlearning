# train.py
import json
import os
from environment import FrozenLakeEnv
from agent import QLearningAgent

def run_training_session(strategy="decaying", episodes=10000, alpha=0.1, gamma=0.95):
    env = FrozenLakeEnv()
    agent = QLearningAgent(
        num_states=env.num_states,
        num_actions=env.num_actions,
        alpha=alpha,
        gamma=gamma,
        epsilon_start=0.2 if strategy == "pure" else 1.0, 
        strategy=strategy
    )

    rewards_history = []
    success_history = []
    epsilon_history = []
    
    window_size = 200
    success_count = 0

    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        done = False

        while not done:
            action = agent.choose_action(state)
            next_state, reward, done = env.step(action)
            agent.update(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

        agent.decay_epsilon()
        
        rewards_history.append(total_reward)
        epsilon_history.append(agent.epsilon)
        
        if total_reward > 0:
            success_count += 1
            success_history.append(1)
        else:
            success_history.append(0)

        # Print telemetry updates periodically
        if (episode + 1) % 2000 == 0:
            recent_success = sum(success_history[-window_size:]) / window_size * 100
            print(f"[{strategy.upper()}] Episode {episode+1}/{episodes} | Recent Success Rate: {recent_success:.1f}% | Epsilon: {agent.epsilon:.3f}")

    return agent.q_table, rewards_history, success_history, epsilon_history, success_count

if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)
    episodes_count = 15000
    
    print("Starting Training Session for Decaying Epsilon-Greedy Strategy...")
    decay_q, decay_rew, decay_succ, decay_eps, decay_total_wins = run_training_session("decaying", episodes=episodes_count)
    
    print("\nStarting Training Session for Pure Epsilon-Greedy Strategy...")
    pure_q, pure_rew, pure_succ, pure_eps, pure_total_wins = run_training_session("pure", episodes=episodes_count)

    # Export analytical payloads safely for evaluation module
    payload = {
        "decay_q": decay_q,
        "pure_q": pure_q,
        "metrics": {
            "decay_rewards": decay_rew,
            "decay_success": decay_succ,
            "decay_epsilon": decay_eps,
            "pure_rewards": pure_rew,
            "pure_success": pure_succ,
            "pure_epsilon": pure_eps
        }
    }
    
    with open("results/training_data.json", "w") as f:
        json.dump(payload, f)
    print("\nTraining complete! Metrics saved to results/training_data.json")
