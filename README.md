# Scene Understanding Based on Text Detections
Humans observing images of a place unknowingly deduct a great deal of information.

Automation of this process requires the algorithm to understand what details are relevant.

It can be assumed that the relevant information for finding a geographic location in the city is present in street signs.

This code was created by undergrads as part of a half-year long project, with the goal to find the geographic location of a place pictured in a set of photo, based on the text present (Assuming the photos can create a panorama).

## Table of Contents
* [Requirements](#requirements)
* [Usage Example](#usage-example)

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


## Sources

@inproceedings{xing2019charnet,
title={Convolutional Character Networks},
author={Xing, Linjie and Tian, Zhi and Huang, Weilin and Scott, Matthew R},
booktitle={Proceedings of the IEEE International Conference on Computer Vision (ICCV)},
year={2019}
}

