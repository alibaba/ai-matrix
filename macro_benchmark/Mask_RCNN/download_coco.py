"""
Script to download the coco dataset, extracted from samples/coco/coco.py.
"""

import os
import sys

import zipfile
import urllib.request
import shutil

def auto_download(dataDir, dataType, dataYear):
    """Download the COCO dataset/annotations if requested.
    dataDir: The root directory of the COCO dataset.
    dataType: What to load (train, val, minival, valminusminival)
    dataYear: What dataset year to load (2014, 2017) as a string, not an integer
    Note:
        For 2014, use "train", "val", "minival", or "valminusminival"
        For 2017, only "train" and "val" annotations are available
    """

    # Setup paths and file names
    if dataType == "minival" or dataType == "valminusminival":
        imgDir = "{}/{}{}".format(dataDir, "val", dataYear)
        imgZipFile = "{}/{}{}.zip".format(dataDir, "val", dataYear)
        imgURL = "http://images.cocodataset.org/zips/{}{}.zip".format("val", dataYear)
    else:
        imgDir = "{}/{}{}".format(dataDir, dataType, dataYear)
        imgZipFile = "{}/{}{}.zip".format(dataDir, dataType, dataYear)
        imgURL = "http://images.cocodataset.org/zips/{}{}.zip".format(dataType, dataYear)
    # print("Image paths:"); print(imgDir); print(imgZipFile); print(imgURL)

    # Create main folder if it doesn't exist yet
    if not os.path.exists(dataDir):
        os.makedirs(dataDir)

    # Download images if not available locally
    if not os.path.exists(imgDir):
        os.makedirs(imgDir)
        print("Downloading images to " + imgZipFile + " ...")
        with urllib.request.urlopen(imgURL) as resp, open(imgZipFile, 'wb') as out:
            shutil.copyfileobj(resp, out)
        print("... done downloading.")
        print("Unzipping " + imgZipFile)
        with zipfile.ZipFile(imgZipFile, "r") as zip_ref:
            zip_ref.extractall(dataDir)
        print("... done unzipping")
    print("Will use images in " + imgDir)

    # Setup annotations data paths
    annDir = "{}/annotations".format(dataDir)
    if dataType == "minival":
        annZipFile = "{}/instances_minival2014.json.zip".format(dataDir)
        annFile = "{}/instances_minival2014.json".format(annDir)
        annURL = "https://dl.dropboxusercontent.com/s/o43o90bna78omob/instances_minival2014.json.zip?dl=0"
        unZipDir = annDir
    elif dataType == "valminusminival":
        annZipFile = "{}/instances_valminusminival2014.json.zip".format(dataDir)
        annFile = "{}/instances_valminusminival2014.json".format(annDir)
        annURL = "https://dl.dropboxusercontent.com/s/s3tw5zcg7395368/instances_valminusminival2014.json.zip?dl=0"
        unZipDir = annDir
    else:
        annZipFile = "{}/annotations_trainval{}.zip".format(dataDir, dataYear)
        annFile = "{}/instances_{}{}.json".format(annDir, dataType, dataYear)
        annURL = "http://images.cocodataset.org/annotations/annotations_trainval{}.zip".format(dataYear)
        unZipDir = dataDir
    # print("Annotations paths:"); print(annDir); print(annFile); print(annZipFile); print(annURL)

    # Download annotations if not available locally
    if not os.path.exists(annDir):
        os.makedirs(annDir)
    if not os.path.exists(annFile):
        if not os.path.exists(annZipFile):
            print("Downloading zipped annotations to " + annZipFile + " ...")
            with urllib.request.urlopen(annURL) as resp, open(annZipFile, 'wb') as out:
                shutil.copyfileobj(resp, out)
            print("... done downloading.")
        print("Unzipping " + annZipFile)
        with zipfile.ZipFile(annZipFile, "r") as zip_ref:
            zip_ref.extractall(unZipDir)
        print("... done unzipping")
    print("Will use annotations in " + annFile)

def main():
    auto_download("./coco_dataset", "train", "2014")
    auto_download("./coco_dataset", "valminusminival", "2014")
    auto_download("./coco_dataset", "minival", "2014")

if __name__ == '__main__':
    main()
