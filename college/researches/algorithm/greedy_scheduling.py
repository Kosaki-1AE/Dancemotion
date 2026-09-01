xs = [(0,2),(1,2),(2,3),(2,5),(3,5),(4,6),(5,7),(6,7),(6,8),(0,7)]
def greedy_scheduling(xs):
    selected = []
    for x in xs:
        for s in selected:
            if (x[0] < s[0] < x[1]) or (x[0] < s[1] < x[1]) \
               or (s[0] < x[0] < s[1]) or (s[0] < x[1] <s[1]):
                break
        else:
            selected.append(x)
    return selected