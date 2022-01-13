import json
from toy_gym.envs.core.workspace_objects import Obstacle, Map, Workspace_objects
# Data to be written
dictionary ={
    "name" : "sathiyajith",
    "rollno" : 56,
    "cgpa" : 8.6,
    "phonenumber" : "9976770500"
}
    
# with open("sample.json", "w") as outfile:
#     json.dump(dictionary, outfile)

# with open("sample.json") as infile:
#     data = json.load(infile)

#print(data['robot'])
Map = Map("Map", [10, 10])
B = Workspace_objects(Map)
A = Obstacle("Body", "Cyclinder", [2,  2], [3,3], True)
print(A.__dict__)
print(B.__dict__)
