import numpy as np
import math

def bezier(var1, var2): #Input arguments: var1 = bezier coefficients, var2 = tau
	t8067 = -1.*var2
	t8079 = 1. + t8067
	p_output1 = math.pow(t8079,5)*var1[0] + 5.*math.pow(t8079,4)*var1[1]*var2 + 10.*math.pow(t8079,3)*var1[2]*math.pow(var2,2) + 10.*math.pow(t8079,2)*var1[3]*math.pow(var2,3) + 5.*t8079*var1[4]*math.pow(var2,4) + var1[5]*math.pow(var2,5)
	return p_output1

def dbezier(var1, var2): #Input arguments: var1 = bezier coefficients, var2 = tau
	t8084 = -1.*var2
	t8085 = 1. + t8084
	t8086 = math.pow(t8085,4)
	t8089 = math.pow(t8085,3)
	t8097 = math.pow(t8085,2)
	t8101 = math.pow(var2,2)
	t8105 = math.pow(var2,3)
	t8111 = math.pow(var2,4)
	p_output1 = -5.*t8086*var1[0] + 5.*t8086*var1[1] - 30.*t8097*t8101*var1[2] + 30.*t8097*t8101*var1[3] - 20.*t8085*t8105*var1[3] + 20.*t8085*t8105*var1[4] - 5.*t8111*var1[4] + 5.*t8111*var1[5] - 20.*t8089*var1[1]*var2 + 20.*t8089*var1[2]*var2
	return p_output1