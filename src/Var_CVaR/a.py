import time

start = time.time()

a = 8000
falta = 7900
lista = []

for i in range(a):
    if i == falta:
        continue
    else:
        lista.append(i)


b = lista
missing = None


for i in range(a):
    missing = i
    for j in b:
        if i == j:
            b.remove(j)
            continue
print(missing)

end = time.time()

print(end-start)


# input
# 1 2 3 4 5 6 7 8 9

# 8 7 5 1 2 6 4 9

# 1st it i=1,
# j


# 1.
# 2. los podria acomodar y luego ver la resta, el primero que me de -1 o 1, uno antes es el que esta desacomodado
