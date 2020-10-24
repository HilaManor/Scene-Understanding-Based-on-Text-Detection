[![Python 3.7.7](https://img.shields.io/badge/python-3.7.7+-blue.svg)](https://www.python.org/downloads/release/python-377/)
[![OpenCV](https://img.shields.io/badge/OpenCV-3.4.2-green)](https://opencv.org/)
[![torch](https://img.shields.io/badge/torch-1.4.0-green)](https://pytorch.org/)
[![torchvision](https://img.shields.io/badge/torchvision-0.5.0-green)](https://pytorch.org/)


# Scene Understanding Based on Text Detections
Humans observing images of a place unknowingly deduct a great deal of information.

Automation of this process requires the algorithm to understand what details are relevant.

It can be assumed that the relevant information for finding a geographic location in the city is present in street signs.

This code was created by undergrads as part of a half-year long project, with the goal to find the geographic location of a place pictured in a set of photo, based on the text present (Assuming the photos can create a panorama).

## Table of Contents
* [Requirements](#requirements)
* [Usage Example](#usage-example)
* [Team](#team)
* [Videos](#videos)
* [The Algorithm](#the-algorithm)
* [Sources](#sources)

## Requirements
The project uses [CharNet](https://github.com/MalongTech/research-charnet) for the text detection. 

- fuzzywuzzy 0.18.0
- python-Levenshtein 0.12.0
- gmplot 1.4.1
- googlemaps 4.4.2
- numpy 1.18.4
### [CharNet](https://github.com/MalongTech/research-charnet) dependancies
- torch 1.4.0
- torchvision 0.5.0
- opencv-python 3.4.2
- opencv-contrib-python 3.4.2
- editdistance 0.5.3
- pyclipper 1.1.0
- shapely 1.7.0
- yacs 0.1.7

If you're having probelms downloading the correct torch/torchvision version, please try using:
```bash
pip install torch===1.4.0 torchvision===0.5.0 -f https://download.pytorch.org/whl/torch_stable.html
```

## Usage Example
To check a signle scene use:
```bash
--single_scene ".\scene" --results_dir ".\output"
```
`scene` should be a folder with a set of photos.

for checking multiple scenes at once, order the images in seperate folders and input the parent folder. e.g.:
```bash 
├── Parent
│   ├── scene 1
│   ├── scene 2
│   ├── scene 3
```
and then use:
```bash
--scenes_dir ".\Parent" --results_dir ".\output"
```

* If you know the photos are ordered by name, you can add the option `--dont_reorder` to vastly improve runtime.
* if `--results_dir` isn't given, the output will be generated in the input directory.

## Team
Hila Manor and Adir Krayden 

Supervised and guided by Elad Hirsch

## Videos
A video showcasing an overview of 3 runs of the projects:
1. A simple scene.
2. A simple scene, yet had problems with Google's GeocodingAPI.
3. A complicated scene.

[![](http://img.youtube.com/vi/mPFdAuVxSTU/0.jpg)](http://www.youtube.com/watch?v=mPFdAuVxSTU "Scene Understanding Based on Text Detections - Overview")

A video showcasing a run that used intersecting locations (close places) to find the location
[![](http://img.youtube.com/vi/VpQQmwEBztc/0.jpg)](http://www.youtube.com/watch?v=VpQQmwEBztc "")

## The Algorithm
1. Panorama Creation
   1. Find images order
      - Random images order input is assumed and fixed
      - Feature-based matching
   2. Estimate focal length
      - based on homographies
   3. Inverse cylindrical warp
      - Use cylindrical panoramas to enable 360° field-of-view.
   4. Stitch panorama
      - Stitch with affine transformation to fix ghosting and drift (camera can be hand-held, and not on a tripod)
2. Signs Extraction
   1. Split the panorama to windows
      - enables runs on weaker GPUs
   2. Extract text using CharNet on each window
      - Splitted words cause a new search in a centered window
      - CharNet “fixes” detected text by comparing to a synthetic dictionary
   3. Concatenate words to signs
      - Match geometry: close words vertically or horizontally
      - Match colors: validate that the backgrounds colors are from the same distribution
   4. Catalogue signs by gradeing similarity to street-signs
      - Background color
      - Keywords presence (e.g. avenue, st.)
      - Appearance in online streets list
   5. Filter out similar variations and long signs
3. Location Search
   1. Query Google’s GeocodeAPI only for street signs
      - This API doesn’t understand points of interest
   2. Query Google’s PlacesAPI for each of the other signs individually.
      - This API can’t handle intersecting data (2 businesses in 1 location)
      - Search for close (geographically) responses
   3. Display options to choose from, and open a marked map

## Sources
- [CharNet](https://github.com/MalongTech/research-charnet)
  - Xing, Linjie and Tian, Zhi and Huang, Weilin and Scott, Matthew R, [“Convolutional Character Networks”](https://arxiv.org/abs/1910.07954), Proceedings of the IEEE International Conference on Computer Vision (ICCV), 2019
- Heung-Yeung Shum and R. Szeliski, Microsoft Research, [“Construction of Panoramic Image Mosaics with Global and Local Alignment”](https://ieeexplore.ieee.org/document/710831), Sixth International Conference on Computer Vision and IJCV, 1999
