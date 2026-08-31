def longest_common_subsequence(str1, str2):
    len_str1 = len(str1)
    len_str2 = len(str2)

    # LCSを格納する表を初期化する
    dp = [[0] * (len_str2 + 1) for _ in range(len_str1 + 1)]

    # LCSを計算する(表の中の数値を表示する場合はこちらにreturn dpを付ける)
    for i in range(1, len_str1 + 1):
        for j in range(1, len_str2 + 1):
            if str1[i - 1] == str2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    #return dp

    # LCSの文字列を復元する
    lcs_length = dp[len_str1][len_str2]
    lcs = [''] * lcs_length
    i = len_str1
    j = len_str2
    while i > 0 and j > 0:
        if str1[i - 1] == str2[j - 1]:
            lcs[lcs_length - 1] = str1[i - 1]
            i -= 1
            j -= 1
            lcs_length -= 1
        elif dp[i - 1][j] > dp[i][j - 1]:
            i -= 1
        else:
            j -= 1

    return ''.join(lcs)

# テスト用の文字列
str1 = "acagcgcaac"
str2 = "ccacgcca"

# LCSを求める
result = longest_common_subsequence(str1, str2)
print("最長共通部分列:", result)

# LCSの表を作成する
table = longest_common_subsequence(str1, str2)

# 結果を出力する
for row in table:
    print(row)