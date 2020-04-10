# project-A

`pip install torch===1.4.0 torchvision===0.5.0 -f https://download.pytorch.org/whl/torch_stable.html`

## dependecies
- torch torchvision (See above)
- opencv-python
- editdistance
- pyclipper
- shapely
- yacs
- matplotlib

## MalongTech / research-charnet
- doesn't have training code (only forward for testing)
  - no baackwards functions
- config file with different params:
```_C.INPUT_SIZE = 2280
_C.SIZE_DIVISIBILITY = 1
_C.WEIGHT= ""

_C.CHAR_DICT_FILE = ""
_C.WORD_LEXICON_PATH = ""

_C.WORD_MIN_SCORE = 0.95
_C.WORD_NMS_IOU_THRESH = 0.15
_C.CHAR_MIN_SCORE = 0.25
_C.CHAR_NMS_IOU_THRESH = 0.3
_C.MAGNITUDE_THRESH = 0.2

_C.WORD_STRIDE = 4
_C.CHAR_STRIDE = 4
_C.NUM_CHAR_CLASSES = 68

_C.WORD_DETECTOR_DILATION = 1
_C.RESULTS_SEPARATOR = chr(31)
```
- `.\tools\test_net.py` returns txt: 4 bounding box coords, TEXT
  - inside charnet function (class overloading) returns char_bboxes, char_scores, word_instances
    - **word_instances**: array of WordInstance: char_scores, text, text_score, word bbox, word_bbox_score
      - word bbox is the entire word polygon coords
    - **char_bboxes**: 2d array of char coords in photo -> in single test we saw 78 chars bboxes but only 74 in the end word bboxes
    - **char_scores**: 2d array: columns = num of chars
- downloaded dataset from [https://rrc.cvc.uab.es/?ch=4]
- קישוטים מפריעים לו
- ignores less than 3 words : charnet.modeling.postprocessing Line 171
  - if only alphabetical letters:
    - =>0.98 V
    - => 0.8 - checks dict
  - else => 0.9 V

---
- Sent mail to malong github + 2nd author. 
  - 2nd author replied, they asked the team manager to publish the code. said that we can contact him if need
### General algorithm system 
- Runs through EAST/TextField (modified) in order to get word bboxes 
- Runs their own characters network (in parallel, unrelated) 
- then it compares word bboxes and characters bboxes, saving every character that overlaps into the word string
  - → This means the word splitting is done by the East/TextField (Need Training code)
  - don't know about the קישוטים problem

# PROBLEM(https://www.google.com/permissions/geoguidelines/)
- street view downloader 360
  - https://istreetview.com/
  
