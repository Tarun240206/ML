num=[2,7,4,1,3,6]
count=0
n=len(num)
for i in range(0,n):
    for j in range(i+1,n):
        if num[i]+num[j]==10:
            count+=1

print("No. of pairs:",count)