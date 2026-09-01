twice = str(input("年収・時給・日給・月給どれが知りたい？:"))
if twice == "年収":
    money = int(input("時給どれくらい？:"))
    time = int(input("何時間？:"))
    day = int(input("週どれくらい？:"))
    Totalmoney1 = money*time*day*365
    print("年収→"+str(Totalmoney1)+"円")
elif twice == "時給":
    money2 = int(input("年収どれくらい？:"))
    workmoney = float(money2/12/30/24)
    day2 = int(input("日数どのくらい？:"))
    overtime = int(input("残業時間どのくらい？:"))
    Totalmoney2 = workmoney/float((day2+overtime))
    print("時給→"+str(Totalmoney2)+"円")
else:
    print("もう一度入力してください")