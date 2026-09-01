count = 0
cnf_dic = {}
for i in range(6):
    x_value = input(f"x{i+1}の値を入力してください:")
    cnf_dic[f'x{i+1}'] = int(x_value)
if str(sum(cnf_dic.values())) <= str(sum(cnf_dic.values())):
    # 論理式を入力
    cnf = input("論理式を入力してください: ")

    if cnf == "not x1" or "not x2" or "not x3" or "not x4" or "not x5":
        # 入力が0なら1に、入力が1なら0に変換
        result = '0' if cnf == '1' else '1'

    if result == 1:
        print("True/充足しました")
        count += 1
    else:
        print("False")
    for i in range(6):
        cnf_dic[f'x{i}'] = '0' if cnf == '1' else '1'