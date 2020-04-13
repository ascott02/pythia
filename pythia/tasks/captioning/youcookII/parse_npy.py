import numpy as np
import sys

data = np.load(sys.argv[1], allow_pickle=True)
# print(type(data))
# for i in range(len(data)):
#     print(i, data[i])
# sys.exit()
for key in data[0].keys():
    print(key, data[0][key])

for i in range(1, len(data)):
    for key in data[i].keys():
        print(key, data[i][key])

sys.exit()

