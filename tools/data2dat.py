# import numpy as np

# with open('group.data', "rb") as datafile:
#     csv_reader = np.loadtxt(datafile, delimiter=',', skiprows=0)
#     array = np.array(csv_reader)

#     with open('knn_datasets.dat', 'wb') as f:
#         for i in array:
#             for j in i:
#                 f.write(np.byte(j))

import numpy as np
import binascii

with open('group.data', "rb") as datafile:
    csv_reader = np.loadtxt(datafile, delimiter=',', skiprows=0)
    array = np.array(csv_reader,'uint8')

    with open('knn_num.dat', 'w') as f:
        count = 0
        for i in array:
            for j in i:
                if count == 3:
                    count = 0
                    j_bin = '{:03b}'.format(j)
                else:
                    count += 1
                    j_bin = '{:08b}'.format(j)
                f.write(str(j_bin))
            f.write("\n")
