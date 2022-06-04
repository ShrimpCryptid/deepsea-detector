import os
import csv

# moves images from IMAGE_FOLDER to NEW_UPLOAD_FOLDER if they are in the ANNOTATED_IMAGE_LIST
# part of the process of extracting only the images that have been manually annotated from Roboflow

IMAGE_FOLDER = "../mate_image_data/"
NEW_UPLOAD_FOLDER = IMAGE_FOLDER + "new_uploads/"
ANNOTATED_IMAGE_LIST = "roboflow_mate_annotated_image_names.csv"

dirname = os.path.dirname(__file__)
filename = os.path.join(dirname, ANNOTATED_IMAGE_LIST)

if not os.path.exists(filename):
  raise FileNotFoundError(ANNOTATED_IMAGE_LIST + " does not exist.")
image_names_file = open(filename, newline='')
image_names_reader = csv.reader(image_names_file)

if (not os.path.isdir(os.path.join(dirname, IMAGE_FOLDER))):
  raise FileNotFoundError(IMAGE_FOLDER + " is not a directory.")

if (not os.path.isdir(os.path.join(dirname, NEW_UPLOAD_FOLDER))):
  raise FileNotFoundError(NEW_UPLOAD_FOLDER + " is not a directory.")

if not os.path.exists(filename):
  raise FileNotFoundError(ANNOTATED_IMAGE_LIST + " does not exist.")
image_names_file = open(filename, newline='')
image_names_reader = csv.reader(image_names_file)

new_uploads = 0;
for row in image_names_reader:
    for val in row:
      name = val.strip()
      path = os.path.join(dirname, IMAGE_FOLDER + name)
      if os.path.exists(path):
        # image in image folder, move to new_uploads
        new_uploads+=1
        new_path = os.path.join(dirname, NEW_UPLOAD_FOLDER + name)
        os.rename(path, new_path) 


print(str(new_uploads) + " new images to upload.")