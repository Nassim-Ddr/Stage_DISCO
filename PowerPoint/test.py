import numpy as np

dtype = [('name', 'S10'), ('height', float), ('age', int)]
values = [('AA', 1.7, 41), 
          ('AB', 1.9, 38),
          ('AA', 1.7, 37),
          ('AC', 1.7, 41),
          ('AB', 1.8, 41),
          ('AB', 2.0, 10),
          ('AA', 1.7, 38),
          ('C', 1, 3)]

a = np.array(values, dtype=dtype)       # create a structured array
values = np.random.shuffle(values)
print("###### Before ########")
for o in a:
    print(o)
a = np.sort(a)
print("###### SORT ########")
for o in a:
    print(o)