import urllib.request
import os
import json

IMAGE_LIST_FILENAME = "fathomnet_fully_verified_non_noaa_image_list.json"
IMAGE_LISTS_FILEPATH = "../data/fathomnet_image_lists/"
IMAGE_DOWNLOAD_FILEPATH = "../data/fathomnet_fully_verified_non_noaa_images/"
DOWNLOAD_LOG_FILENAME = "000_downloaded_imgs_log.txt"
SKIPPED_LOG_FILENAME = "000_skipped_imgs_log.txt"

def get_filename_from_url(url):
    return url.split('/').pop()

dirname = os.path.dirname(__file__)

filename = os.path.join(dirname, IMAGE_DOWNLOAD_FILEPATH)
if not os.path.exists(filename):
    raise FileNotFoundError(filename + " does not exist.")

filename = os.path.join(dirname, IMAGE_LISTS_FILEPATH + IMAGE_LIST_FILENAME)
if not os.path.exists(filename):
    raise FileNotFoundError(filename + " does not exist.")
image_list_file = open(filename)
image_list = json.load(image_list_file)

filename = os.path.join(dirname, IMAGE_DOWNLOAD_FILEPATH + DOWNLOAD_LOG_FILENAME)
if os.path.exists(filename):
    raise FileExistsError(filename + " already exists, delete or rename")
download_log = open(filename, 'w')

filename = os.path.join(dirname, IMAGE_DOWNLOAD_FILEPATH + SKIPPED_LOG_FILENAME)
if os.path.exists(filename):
    raise FileExistsError(filename + " already exists, delete or rename")
skipped_log = open(filename, 'w')



for img in image_list:
    url = img["url"]
    name = get_filename_from_url(url)
    filepath = os.path.join(dirname, IMAGE_DOWNLOAD_FILEPATH + name)
    if not os.path.exists(filepath):
        urllib.request.urlretrieve(url, filepath)
        print(url)
        download_log.write(url + '\n')
    else:
        print("SKIPPED: " + url)
        skipped_log.write(url + '\n')
