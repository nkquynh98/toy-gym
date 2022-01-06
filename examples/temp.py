import numpy as np

dot = -0.5
arc = np.arccos(dot)
print(arc)

def calculate_angle(vector):
    vector2 = np.array([1,0])
    vector1 = vector/np.linalg.norm(vector)
    vector2 = vector2/np.linalg.norm(vector2)
    dot = np.dot(vector1, vector2)
    print("Arccos", np.arccos(dot))
    arctan = np.arctan2(vector[1], vector[0])
    print("Arctan", arctan)

test_vector = np.array([1,1])
calculate_angle(test_vector)
test_vector = np.array([-2,-2])
calculate_angle(test_vector)