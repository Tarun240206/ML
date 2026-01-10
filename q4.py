str=input("enter the word:").lower()
freq={}

for i in str:
  if i.isalpha():
    freq[i] = freq.get(i, 0) + 1

max_ch=""
max_freq=0

for i in freq:
  if freq[i]>max_freq:
    max_freq=freq[i]
    max_ch=i

print(max_ch)
print(max_freq)