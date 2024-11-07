import random
from collections import defaultdict
import matplotlib.pyplot as plt
import json
from tqdm import tqdm

# Define valid card options
valid_cards = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'Jack', 'Queen', 'King', 'Ace']

def random_card_draw():
    """Randomly draw a card from the deck."""
    return random.choice(valid_cards)

def calculate_hand_value(hand):
    """Calculate the total value of a hand, treating Aces as 1 or 11 to optimize hand value."""
    value = 0
    aces = 0
    for card in hand:
        if card in ['Jack', 'Queen', 'King']:
            value += 10
        elif card == 'Ace':
            aces += 1
            value += 11  # Initially count Ace as 11
        else:
            value += int(card)
    while value > 21 and aces:
        value -= 10  # Convert an Ace from 11 to 1
        aces -= 1
    return value

def player_strategy(dealer_upcard, player_hand, player_card_frequencies, dealer_hand):
    player_hand_value = calculate_hand_value(player_hand)
    while player_hand_value < 21:
        if dealer_upcard in ['Jack', 'Queen', 'King', '10'] and player_hand_value < 17:
            card = random_card_draw()
            player_hand.append(card)
            player_card_frequencies[card] += 1
            player_hand_value = calculate_hand_value(player_hand)
            if player_hand_value > 21:
                return player_hand_value
        elif dealer_upcard in range(2, 7) and player_hand_value < 12:
            card = random_card_draw()
            player_hand.append(card)
            player_card_frequencies[card] += 1
            player_hand_value = calculate_hand_value(player_hand)
            if player_hand_value > 21:
                return player_hand_value
        else:
            break
    return player_hand_value

dealer_bust_count = 0

def dealer_strategy(dealer_hand, dealer_card_frequencies, player_hand):
    dealer_hand_value = calculate_hand_value(dealer_hand)
    dealer_busted = False
    while dealer_hand_value < 17 or (dealer_hand_value == 17 and 'Ace' in dealer_hand):
        card = random_card_draw()
        dealer_hand.append(card)
        dealer_card_frequencies[card] += 1
        dealer_hand_value = calculate_hand_value(dealer_hand)
    if dealer_hand_value > 21:
        global dealer_bust_count
        dealer_bust_count += 1
        dealer_busted = True
    return dealer_hand_value, dealer_busted

def declare_winner(player_hand_value, dealer_hand_value):
    if player_hand_value > 21:
        return "Dealer wins! Player busted."
    elif dealer_hand_value > 21:
        return "Player wins! Dealer busted."
    elif player_hand_value > dealer_hand_value:
        return "Player wins!"
    elif dealer_hand_value > player_hand_value:
        return "Dealer wins!"
    else:
        return "It's a tie!"

# Lists to store final hand values and card frequencies for each round
player_final_hand_values = []
dealer_final_hand_values = []
player_card_frequencies = defaultdict(int)
dealer_card_frequencies = defaultdict(int)
win_record = defaultdict(int, {"Player": 0, "Dealer": 0, "Tie": 0})

# Function to run the game once and record results
def run_single_game():
    dealer_hand = [random_card_draw(), random_card_draw()]
    dealer_upcard = dealer_hand[0]
    
    for card in dealer_hand:
        dealer_card_frequencies[card] += 1
    
    player_hand = [random_card_draw(), random_card_draw()]
    
    for card in player_hand:
        player_card_frequencies[card] += 1

    player_hand_value = player_strategy(dealer_upcard, player_hand, player_card_frequencies, dealer_hand)
    
    if player_hand_value <= 21:
        dealer_hand_value, dealer_busted = dealer_strategy(dealer_hand, dealer_card_frequencies, player_hand)
    else:
        dealer_hand_value = calculate_hand_value(dealer_hand)

    player_final_hand_values.append(player_hand_value)
    dealer_final_hand_values.append(dealer_hand_value)
    
    result = declare_winner(player_hand_value, dealer_hand_value)
    if "Player wins" in result:
        win_record['Player'] += 1
    elif "Dealer wins" in result:
        win_record['Dealer'] += 1
    else:
        win_record['Tie'] += 1

    print("Game Details:")
    print(f"  Dealer Hand: {dealer_hand}, Dealer Hand Value: {dealer_hand_value}")
    print(f"  Player Hand: {player_hand}, Player Hand Value: {player_hand_value}")
    print(f"  Result: {result}\n")

dealer_bust_count = 0
for _ in tqdm(range(1000), desc="Running games"):
    run_single_game()

dealer_bust_rate = (dealer_bust_count / 1000) * 100
player_win_rate = (win_record['Player'] / 1000) * 100
average_player_hand_value = sum(player_final_hand_values) / len(player_final_hand_values)
average_dealer_hand_value = sum(dealer_final_hand_values) / len(dealer_final_hand_values)

print(f"Randomized Blackjack - Player Win Rate: {player_win_rate:.3f}%")
print(f"Randomized Blackjack - Dealer Bust Rate: {dealer_bust_rate:.3f}%")
print(f"Randomized Blackjack - Average Player Hand Value: {average_player_hand_value:.2f}")
print(f"Randomized Blackjack - Average Dealer Hand Value: {average_dealer_hand_value:.2f}")

results = {
    "player_card_frequencies": dict(player_card_frequencies),
    "dealer_card_frequencies": dict(dealer_card_frequencies),
    "player_final_hand_values": player_final_hand_values,
    "dealer_final_hand_values": dealer_final_hand_values,
    "win_record": dict(win_record),
    "metrics": {
        "player_win_rate": player_win_rate,
        "dealer_bust_rate": dealer_bust_rate,
        "average_player_hand_value": average_player_hand_value,
        "average_dealer_hand_value": average_dealer_hand_value
    }
}

filename = "random_blackjack_results.json"
with open(filename, "w") as file:
    json.dump(results, file)

# Define color schemes
player_color = 'blue'
dealer_color = 'orange'

plt.figure(figsize=(10, 5))
plt.hist(player_final_hand_values, bins=range(10, 31), alpha=0.5, label='Player Hand Values', color=player_color)
plt.hist(dealer_final_hand_values, bins=range(10, 31), alpha=0.5, label='Dealer Hand Values', color=dealer_color)
plt.xlabel('Final Hand Value')
plt.ylabel('Frequency')
plt.legend()
plt.title('Distribution of Final Hand Values over 1000 Games (Randomized Blackjack)')
plt.show()

fig, ax = plt.subplots(1, 2, figsize=(14, 5))

ax[0].bar(player_card_frequencies.keys(), player_card_frequencies.values(), color=player_color)
ax[0].set_title('Player Card Draw Frequencies (Randomized Blackjack)')
ax[0].set_xlabel('Card')
ax[0].set_ylabel('Frequency')

ax[1].bar(dealer_card_frequencies.keys(), dealer_card_frequencies.values(), color=dealer_color)
ax[1].set_title('Dealer Card Draw Frequencies (Randomized Blackjack)')
ax[1].set_xlabel('Card')
ax[1].set_ylabel('Frequency')

plt.suptitle('Card Frequencies (Randomized Blackjack)')
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()

labels = list(win_record.keys())
sizes = list(win_record.values())

plt.figure(figsize=(7, 7))
plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140)
plt.title('Win Rate Over 1000 Games (Randomized Blackjack)')
plt.show()
