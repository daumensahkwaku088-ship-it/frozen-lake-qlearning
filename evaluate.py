# evaluate.py
import json
import os
from environment import FrozenLakeEnv

def extract_and_display_policy(q_table, env):
    """Part D: Policy Extraction and Console Printing Grid."""
    symbols = {0: "←", 1: "↓", 2: "→", 3: "↑"}
    print("\nExtracted Grid Policy Map Layout:")
    print("=" * 33)
    
    for r in range(env.rows):
        row_str = "| "
        for c in range(env.cols):
            state = r * env.cols + c
            cell = env.map[r][c]
            
            if cell == "H":
                row_str += "H   | "
            elif cell == "G":
                row_str += "G   | "
            else:
                # Best action based on values
                best_action = q_table[state].index(max(q_table[state]))
                # show default question mark placeholder If unvisited
                if max(q_table[state]) == 0.0:
                    row_str += "?   | "
                else:
                    row_str += f"{symbols[best_action]}   | "
        print(row_str)
        print("-" * 33)

def evaluate_policy(q_table, env, runs=100):
    """Part E: Evaluates the performance metrics over test episodes."""
    successful_runs = 0
    failures = 0
    total_rewards = 0.0

    for _ in range(runs):
        state = env.reset()
        done = False
        while not done:
            action = q_table[state].index(max(q_table[state]))
            state, reward, done = env.step(action)
        
        total_rewards += reward
        if reward > 0.0:
            successful_runs += 1
        else:
            failures += 1

    success_rate = (successful_runs / runs) * 100
    avg_reward = total_rewards / runs
    
    return success_rate, avg_reward, failures, successful_runs

def generate_ascii_chart(decay_succ, pure_succ, window=1000):
    """Bonus Task B: Generates textual visualization line data metrics."""
    chart_lines = ["\n=== TRAINING PERFORMANCE PLOT GRAPH COMPARISON ==="]
    chart_lines.append(f"Aggregated Performance blocks checked every {window} runs:\n")
    chart_lines.append(f"{'Block Range':<15} | {'Decaying Epsilon Success':<25} | {'Pure Epsilon Success':<25}")
    chart_lines.append("-" * 73)
    
    for idx in range(0, len(decay_succ), window):
        slice_d = decay_succ[idx:idx+window]
        slice_p = pure_succ[idx:idx+window]
        rate_d = (sum(slice_d) / len(slice_d)) * 100 if slice_d else 0
        rate_p = (sum(slice_p) / len(slice_p)) * 100 if slice_p else 0
        
        # Generation of ASCII visual bar
        bar_d = "#" * int(rate_d / 10)
        chart_lines.append(f"Run {idx:05d}-{idx+window:05d} | {rate_d:>5.1f}% {bar_d:<10} | {rate_p:>5.1f}%")
        
    return "\n".join(chart_lines)

if __name__ == "__main__":
    if not os.path.exists("results/training_data.json"):
        print("Error: training_data.json not found. Run train.py first.")
        exit(1)
        
    with open("results/training_data.json", "r") as f:
        data = json.load(f)

    env = FrozenLakeEnv()
    
    print("\n" + "#"*40)
    print(" EVALUATION RESULTS: DECAYING EPSILON AGENT ")
    print("#"*40)
    
    d_rate, d_avg, d_fail, d_win = evaluate_policy(data["decay_q"], env)
    print(f"Success Rate (%)   : {d_rate:.2f}%")
    print(f"Average Reward     : {d_avg:.2f}")
    print(f"Number of Failures : {d_fail}")
    print(f"Successful Runs    : {d_win}")
    extract_and_display_policy(data["decay_q"], env)

    print("\n" + "#"*40)
    print(" EVALUATION RESULTS: PURE EPSILON AGENT ")
    print("#"*40)
    
    p_rate, p_avg, p_fail, p_win = evaluate_policy(data["pure_q"], env)
    print(f"Success Rate (%)   : {p_rate:.2f}%")
    print(f"Average Reward     : {p_avg:.2f}")
    print(f"Number of Failures : {p_fail}")
    print(f"Successful Runs    : {p_win}")
    
    # Saving ASCII charts to file
    ascii_graph = generate_ascii_chart(data["metrics"]["decay_success"], data["metrics"]["pure_success"])
    with open("results/performance_chart.txt", "w") as f:
        f.write(ascii_graph)
    
    print("\nVisual performance graph written successfully to: results/performance_chart.txt")
