def max_slice_1(xs):
    max_i, max_j, max_slice = 0, 1, xs[0]
    for i in range(len(xs)):
        tmp = 0
        for j in range(i, len(xs)):
            tmp += xs[j]
            if tmp > max_slice:
                max_i, max_j, max_slice = i, j + 1, tmp  # 最後のインデックス +1 を max_j として設定
    return max_i, max_j, max_slice

def max_slice_dc(xs):
    if len(xs) <= 1:
        return 0, 1, xs[0] if xs else 0

    mid = len(xs) // 2

    # 左半分の最大部分配列
    left_i, left_j, left_slice = max_slice_dc(xs[:mid])

    # 右半分の最大部分配列
    right_i, right_j, right_slice = max_slice_dc(xs[mid:])

    # 中央をまたぐ最大部分配列
    max_i, max_j, max_slice = mid, mid + 1, 0
    tmp_left, tmp_right = 0, 0

    for i in range(mid - 1, -1, -1):
        tmp_left += xs[i]
        if tmp_left > max_slice:
            max_i, max_slice = i, tmp_left

    for j in range(mid, len(xs)):
        tmp_right += xs[j]
        if tmp_right > max_slice:
            max_j, max_slice = j + 1, tmp_right  # 最後のインデックス +1 を max_j として設定

    # 左半分、右半分、中央をまたぐ部分のうち最大のものを選択
    if left_slice >= right_slice and left_slice >= max_slice:
        return left_i, left_j, left_slice
    elif right_slice >= left_slice and right_slice >= max_slice:
        return right_i + mid, right_j + mid, right_slice
    else:
        return max_i, max_j, max_slice
