import numpy as np

position = np.array([[-0.2000, 0.7546, 0.1675, 2.4948, 0.4405, 3.1894, 0.2132],
[-0.1484, 0.7460, 0.1734, 2.4568, 0.6307, 2.9492, 0.6271],
[-0.1299, 0.7518, 0.1767, 2.4873, 0.6133, 2.9434, 0.6740],
[-0.1121, 0.7567, 0.1794, 2.5177, 0.5954, 2.9297, 0.7256]])

velocity = np.array([[0.7743, 0.2891, 0.3796, 1.1377, -0.9273, -0.1285, 1.6298],
                     [0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0]])

a_rightS_aux = np.array([0.3310, -0.6036, 0.3061, -0.0033, -0.3794, 0.4163, 
0.3419, -0.8885, 0.4993, 0.0141, -0.6895, 0.6282, 
0.3465, -0.8171, 0.4639, 0.5166, -1.4829, 0.9596, 
0.0126, 0.0755, 0.0867, 0.5326, -0.7967, 0.4389, 
-0.0139, -0.0732, 0.2052, 0.3393, -0.3877, 0.1665, 
-0.0033, -0.3794, 0.4163, 0.3310, -0.6036, 0.3061])

a_rightS = a_rightS_aux.reshape(6,6)

a_leftS_aux = np.array([a_rightS_aux[3], a_rightS_aux[4], a_rightS_aux[5], a_rightS_aux[0], a_rightS_aux[1], a_rightS_aux[2],
                a_rightS_aux[9], a_rightS_aux[10], a_rightS_aux[11], a_rightS_aux[6], a_rightS_aux[7], a_rightS_aux[8],
                a_rightS_aux[15], a_rightS_aux[16], a_rightS_aux[17], a_rightS_aux[12], a_rightS_aux[13], a_rightS_aux[14],
                a_rightS_aux[21], a_rightS_aux[22], a_rightS_aux[23], a_rightS_aux[18], a_rightS_aux[19], a_rightS_aux[20],
                a_rightS_aux[27], a_rightS_aux[28], a_rightS_aux[29], a_rightS_aux[24], a_rightS_aux[25], a_rightS_aux[26],
                a_rightS_aux[33], a_rightS_aux[34], a_rightS_aux[35], a_rightS_aux[30], a_rightS_aux[31], a_rightS_aux[32]])

a_leftS = a_leftS_aux.reshape(6,6)

p = np.array([0.5, 0])

q = np.array([-0.2000, 0.7546, 0.1675, 2.4948, 0.4405, 3.1894, 0.2132])

