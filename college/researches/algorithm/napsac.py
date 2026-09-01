def knapsack(items, capacity):
    n = len(items)
    # dp[i][j]はi番目のアイテムまで考慮して、重さがj以下のときの最大価値を表す
    dp = [[0] * (capacity + 1) for _ in range(n + 1)]

    for i in range(1, n + 1):
        for w in range(capacity + 1):
            # i番目のアイテムを選べる場合と選べない場合で価値の最大値を選択
            if items[i - 1][1] <= w:
                dp[i][w] = max(dp[i - 1][w], dp[i - 1][w - items[i - 1][1]] + items[i - 1][2])
            else:
                dp[i][w] = dp[i - 1][w]

    # 最適解の価値を取得
    optimal_value = dp[n][capacity]
   
    # 選択されたアイテムを特定
    selected_items = []
    w = capacity
    for i in range(n, 0, -1):
        if dp[i][w] != dp[i - 1][w]:
            selected_items.append(items[i - 1])
            w -= items[i - 1][1]

    return optimal_value, selected_items

# テスト用例
items = [(0, 360, 576), (1, 250, 375), (2, 220, 264), (3, 340, 714), (4, 400, 720), (5, 200, 260), (6, 240, 408), (7, 280, 560), (8, 260, 286), (9, 350, 665)]
capacity = 1800
result_value, result_items = knapsack(items, capacity)
print("最適な価値:", result_value)
print("選択されたアイテム:", result_items)