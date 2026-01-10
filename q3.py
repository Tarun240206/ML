import numpy as np

n = int(input("Enter size of square matrix (n): "))

M = []
for i in range(n):
    row = []
    for j in range(n):
        row.append(int(input(f"Enter M[{i}][{j}] value: ")))
    M.append(row)

M = np.array(M)

m = int(input("Enter power m: "))

result = np.linalg.matrix_power(M, m)

print(f"\nMatrix M^{m}:")
print(result)