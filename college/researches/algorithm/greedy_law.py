class Item:
    def __init__(self, weight, value):
        self.weight = weight
        self.value = value

def knapsack_greedy(items, capacity, criterion):
    if criterion == "value":
        items.sort(key=lambda x: x.value, reverse=True)
    elif criterion == "value_per_weight":
        items.sort(key=lambda x: x.value / x.weight, reverse=True)
    elif criterion == "weight":
        items.sort(key=lambda x: x.weight)

    knapsack = []
    total_weight = 0
    total_value = 0

    for item in items:
        if total_weight + item.weight <= capacity:
            knapsack.append(item)
            total_weight += item.weight
            total_value += item.value

    return knapsack, total_weight, total_value

# アイテムのリストを作成する
items = [
    Item(5, 10),  # (重さ, 価値)
    Item(8, 15),
    Item(6, 8),
    Item(3, 6),
    Item(4, 7),
    Item(7, 12),
    Item(10, 20),
    Item(2, 4),
    Item(9, 18),
    Item(1, 2)
]

# ナップサックの容量と選択基準を設定する
capacity = 30  # ナップサックの容量
criteria = ["value", "value_per_weight", "weight"]

for criterion in criteria:
    selected_items, total_weight, total_value = knapsack_greedy(items, capacity, criterion)
    print(f"選択基準: {criterion}")
    print("選択されたアイテム:")
    for item in selected_items:
        print(f"重さ: {item.weight}, 価値: {item.value}")
    print(f"合計重さ: {total_weight}, 合計価値: {total_value}")
    print("-------------------------")