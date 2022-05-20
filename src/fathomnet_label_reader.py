import json
import os


dirname = os.path.dirname(__file__)
filename = os.path.join(dirname, '../../data/fathomnet_global.json')
fathomnet_file = open(filename)
fathomnet_data = json.load(fathomnet_file)

for i in fathomnet_data:
  print(i)
  break

