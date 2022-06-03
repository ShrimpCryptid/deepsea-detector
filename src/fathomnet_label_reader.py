import json
import os

DATA_FILES_PREFIX = "../data/"
OUTPUT_FOLDER_PREFIX = "fathomnet_output/"
FATHOMNET_DATA_FILE = "fathomnet_global.json"

FULLY_VERIFIED_PREFIX = "fully_verified"
PARTIALLY_VERIFIED_PREFIX = "partially_verfied"
NOAA_PREFIX = "noaa"
NON_NOAA_PREFIX = "non_noaa"
PARTIALLY_VERIFIED_PHOTO_LIST_FILE = "fathomnet_partially_verified.csv"
FULLY_VERIFIED_PHOTO_LIST_FILE = "fathomnet_fully_verified.csv"
PARTIALLY_VERIFIED_PHOTO_LIST_FILE = "fathomnet_partially_verified.csv"


dirname = os.path.dirname(__file__)
filename = os.path.join(dirname, DATA_FILES_PREFIX + FATHOMNET_DATA_FILE)

if not os.path.exists(filename):
  raise FileNotFoundError(filename + " does not exist.")
fathomnet_file = open(filename)
fathomnet_data = json.load(fathomnet_file)
'''
filename = os.path.join(dirname, DATA_FILES_PREFIX + OUTPUT_FOLDER_PREFIX + FULLY_VERIFIED_PHOTO_LIST_FILE)
if os.path.exists(filename):
  raise FileExistsError(filename + " already exists, please rename or delete")
fully_verified_photos = open(filename)

filename = os.path.join(dirname, DATA_FILES_PREFIX + OUTPUT_FOLDER_PREFIX + PARTIALLY_VERIFIED_PHOTO_LIST_FILE)
if os.path.exists(filename):
  raise FileExistsError(filename + " already exists, please rename or delete")
partially_verified_photos = open(filename)
'''
noaa_count = 0
non_noaa_count = 0
for i in fathomnet_data:
  if "boundingBoxes" in i:
    bounding_info_arr = i["boundingBoxes"]
    at_least_one_verified = False
    all_verified = True
    for b in bounding_info_arr:
      if b["verified"] == True:
        at_least_one_verified = True
      else:
        all_verified = False

  
  if (at_least_one_verified):
    if ("noaa.gov" in i["url"]): 
      noaa_count+=1
      print(i["id"])
    else:
      non_noaa_count+=1
  
print()
print(noaa_count)
print(non_noaa_count)
  

