import time

import python

# Genesys Phase: 思考
a = 10
b = 20

print(f"Python: {a} と {b} をアセンブリに渡します...")

# Motion Phase: 実行
# この瞬間、処理はCを経由してCPUのレジスタで直接演算される
start_time = time.perf_counter()
result = python.add_asm(a, b)
end_time = time.perf_counter()

# Coherence Phase: 結果確認
print(f"Assembly Result: {result}")
print(f"Time: {end_time - start_time:.9f} sec")

if result == 30:
    print("成功：思考と物理動作が直結しました。")
else:
    print("失敗：ノイズが混じっています。")