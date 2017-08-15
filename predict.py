import numpy as np

N = 1000
S0 = "0"
S1 = "01"
while True:
    S = S1 + S0
    S0 = S1
    S1 = S
    if len(S) > N:
        S = S[:N]
        break
print(S, len(S))
data = [int(s) for s in S]
print(data)
print(len(data))
