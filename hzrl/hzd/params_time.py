import numpy as np

position = np.array([[-0.2000, 0.7546, 0.1675, 2.4948, 0.4405, 3.1894, 0.2132],
[-0.1484, 0.7460, 0.1734, 2.4568, 0.6307, 2.9492, 0.6271],
[-0.1299, 0.7518, 0.1767, 2.4873, 0.6133, 2.9434, 0.6740],
[-0.1121, 0.7567, 0.1794, 2.5177, 0.5954, 2.9297, 0.7256]])

velocity = np.array([[0.7743, 0.2891, 0.3796, 1.1377, -0.9273, -0.1285, 1.6298],
                     [0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0]])



a_rightS_aux = np.array([2.4950, 0.4402, 3.1895, 0.2130, 2.6278, 0.3316, 
3.1746, 0.4026, 2.8602, 0.1096, 2.9946, 1.0159, 
3.0932, 0.0616, 2.5419, 1.1197, 3.1368, 0.0565, 
2.4437, 0.7600, 3.1895, 0.2130, 2.4950, 0.4401])

a_rightS = a_rightS_aux.reshape(6,4)

a_leftS_aux = np.array([a_rightS_aux[2], a_rightS_aux[3], a_rightS_aux[0], a_rightS_aux[1],
                a_rightS_aux[6], a_rightS_aux[7], a_rightS_aux[4], a_rightS_aux[5],
                a_rightS_aux[10], a_rightS_aux[11], a_rightS_aux[8], a_rightS_aux[9],
                a_rightS_aux[14], a_rightS_aux[15], a_rightS_aux[12], a_rightS_aux[13],
                a_rightS_aux[18], a_rightS_aux[19], a_rightS_aux[16], a_rightS_aux[17],
                a_rightS_aux[22], a_rightS_aux[23], a_rightS_aux[20], a_rightS_aux[21]])

a_leftS = a_leftS_aux.reshape(6,4)

p = np.array([0.5, 0])

q = np.array([-0.2000, 0.7546, 0.1675, 2.4948, 0.4405, 3.1894, 0.2132])

