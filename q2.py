l = list(map(int, input().split()))
n = len(l)

if n < 2:
    print("Range determination not possible")
else:
    max_v = l[0]
    min_v = l[0]

    for i in range(n):
        if l[i] < min_v:
            min_v = l[i]
        if l[i] > max_v:
            max_v = l[i]

    diff = max_v - min_v
    print(diff)