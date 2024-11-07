import json

files = {
    "baseline_data": "random_blackjack_results.json",
    "llama3.1_fewshot_temp_0.5_deceptive": "llama3.1_fewshot_temp_0.5_deceptive.json",
    "llama3.1_fewshot_temp0.5": "llama3.1_fewshot_temp0.5.json"
}

def load_and_analyze_json(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)

    print(f"Columns in {file_path.split('/')[-1]}:")
    for key in data.keys():
        print(f" - {key}")

    print("\nSample data from each column:")
    for key, value in data.items():
        print(f"Column: {key}")
        print(f" - Type: {type(value)}")
        # Display a sample of data
        if isinstance(value, dict):  # Card frequencies
            print(f" - Sample data (first 3 items): {dict(list(value.items())[:3])}")
        elif isinstance(value, list):  # Final hand values
            print(f" - Sample data (first 5 items): {value[:5]}")
        else:
            print(f" - Value: {value}")
        print()

for file_name, file_path in files.items():
    print(f"\nAnalyzing file: {file_name}")
    load_and_analyze_json(file_path)

import json
import numpy as np
from scipy.special import kl_div

# Load JSON data
def load_json_data(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

# Normalize frequencies to form probability distributions
def normalize_frequencies(frequencies):
    total = sum(frequencies.values())
    return {key: count / total for key, count in frequencies.items()}

# Calculate KL Divergence between two distributions
def calculate_kl_divergence(dist1, dist2, epsilon=1e-10):
    # Ensure both distributions have the same keys by filling missing values with epsilon
    all_keys = set(dist1.keys()).union(set(dist2.keys()))
    p = np.array([dist1.get(key, epsilon) for key in all_keys])
    q = np.array([dist2.get(key, epsilon) for key in all_keys])

    # Normalize distributions to ensure they sum to 1
    p /= p.sum()
    q /= q.sum()

    # Calculate KL Divergence using scipy's kl_div function
    return np.sum(kl_div(p, q))

# Load data
baseline_data = load_json_data(files["baseline_data"])
deceptive_data = load_json_data(files["llama3.1_fewshot_temp_0.5_deceptive"])
fewshot_data = load_json_data(files["llama3.1_fewshot_temp0.5"])

# Normalize player and dealer frequencies
baseline_player = normalize_frequencies(baseline_data["player_card_frequencies"])
baseline_dealer = normalize_frequencies(baseline_data["dealer_card_frequencies"])

deceptive_player = normalize_frequencies(deceptive_data["player_card_frequencies"])
deceptive_dealer = normalize_frequencies(deceptive_data["dealer_card_frequencies"])

fewshot_player = normalize_frequencies(fewshot_data["player_card_frequencies"])
fewshot_dealer = normalize_frequencies(fewshot_data["dealer_card_frequencies"])

# Calculate KL Divergence for each comparison
print("KL Divergence Results:")

# 1. Baseline vs Fewshot
kl_player_baseline_fewshot = calculate_kl_divergence(baseline_player, fewshot_player)
kl_dealer_baseline_fewshot = calculate_kl_divergence(baseline_dealer, fewshot_dealer)
print(f"1. Baseline vs Non-Deceptive- Player: {kl_player_baseline_fewshot}, Dealer: {kl_dealer_baseline_fewshot}")

# 2. Baseline vs Deceptive
kl_player_baseline_deceptive = calculate_kl_divergence(baseline_player, deceptive_player)
kl_dealer_baseline_deceptive = calculate_kl_divergence(baseline_dealer, deceptive_dealer)
print(f"2. Baseline vs Deceptive - Player: {kl_player_baseline_deceptive}, Dealer: {kl_dealer_baseline_deceptive}")

# 3. Deceptive vs Fewshot
kl_player_deceptive_fewshot = calculate_kl_divergence(deceptive_player, fewshot_player)
kl_dealer_deceptive_fewshot = calculate_kl_divergence(deceptive_dealer, fewshot_dealer)
print(f"3. Deceptive vs Non-Deceptive - Player: {kl_player_deceptive_fewshot}, Dealer: {kl_dealer_deceptive_fewshot}")

# Convert list of values into frequency distribution
def list_to_frequency_distribution(value_list):
    freq_dict = {}
    for value in value_list:
        freq_dict[value] = freq_dict.get(value, 0) + 1
    return freq_dict

# Convert final hand values into frequency distributions
baseline_player_final = normalize_frequencies(list_to_frequency_distribution(baseline_data["player_final_hand_values"]))
baseline_dealer_final = normalize_frequencies(list_to_frequency_distribution(baseline_data["dealer_final_hand_values"]))

deceptive_player_final = normalize_frequencies(list_to_frequency_distribution(deceptive_data["player_final_hand_values"]))
deceptive_dealer_final = normalize_frequencies(list_to_frequency_distribution(deceptive_data["dealer_final_hand_values"]))

fewshot_player_final = normalize_frequencies(list_to_frequency_distribution(fewshot_data["player_final_hand_values"]))
fewshot_dealer_final = normalize_frequencies(list_to_frequency_distribution(fewshot_data["dealer_final_hand_values"]))

# Calculate KL Divergence for each comparison of final hand values
# 1. Baseline vs Fewshot
kl_player_baseline_fewshot_final = calculate_kl_divergence(baseline_player_final, fewshot_player_final)
kl_dealer_baseline_fewshot_final = calculate_kl_divergence(baseline_dealer_final, fewshot_dealer_final)

# 2. Baseline vs Deceptive
kl_player_baseline_deceptive_final = calculate_kl_divergence(baseline_player_final, deceptive_player_final)
kl_dealer_baseline_deceptive_final = calculate_kl_divergence(baseline_dealer_final, deceptive_dealer_final)

# 3. Deceptive vs Fewshot
kl_player_deceptive_fewshot_final = calculate_kl_divergence(deceptive_player_final, fewshot_player_final)
kl_dealer_deceptive_fewshot_final = calculate_kl_divergence(deceptive_dealer_final, fewshot_dealer_final)

# Storing the results
kl_player_baseline_fewshot_final, kl_dealer_baseline_fewshot_final, kl_player_baseline_deceptive_final, kl_dealer_baseline_deceptive_final, kl_player_deceptive_fewshot_final, kl_dealer_deceptive_fewshot_final

# Printing the KL divergence values with descriptive sentences

print("KL Divergence Results for Final Hand Values:\n")

# 1. Baseline vs Fewshot
print(f"1. Baseline vs Non-Deceptive Dealer - Player Final: {kl_player_baseline_fewshot_final:.4f}, Dealer Final: {kl_dealer_baseline_fewshot_final:.4f}")

# 2. Baseline vs Deceptive
print(f"2. Baseline vs Deceptive - Player Final: {kl_player_baseline_deceptive_final:.4f}, Dealer Final: {kl_dealer_baseline_deceptive_final:.4f}")

# 3. Deceptive vs Fewshot
print(f"3. Deceptive vs Non-Deceptive Dealer - Player Final: {kl_player_deceptive_fewshot_final:.4f}, Dealer Final: {kl_dealer_deceptive_fewshot_final:.4f}")