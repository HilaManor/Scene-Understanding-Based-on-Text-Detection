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

# [PROBLEM](https://www.google.com/permissions/geoguidelines/)
- street view downloader 360
  - https://istreetview.com/
  

 Issues -
  1. Dictionary - check with Elad if changes are needed
     - 'AAA'
     when final word score is 80%-98%, using the dict for word recog - creating false words.
     -  KERE - maybe change the score parameter / when to use dict
        **Even when only 1 word had decreased the score (k)**
     -  when finding only a part of a word - using dict (false word) 
        (*B*LOND - LOND)
     
  2. Less than 3 characters - (st)      
  
  3. corners 
     - creating false words
     - 
     
  4. Size Of Pic - 
     - Panorama    
     
  5. Colors Changes - 
     - when a word is written with more than 1 color (Loss of 'H' - half black half white - New Zealand)  
     - background (Barny Martin - Australia)  
       
  6. HyperParameters (?) -
     - char_min_score & Word_min_score
     - input size - Require strong GPU (recognize more in panorama pic): computer lab? helpful?
     
  A. Various pics from different angles of the same word provides different recognition (if the word was cut / in the corner in one of them Vs presented as whole word in another pic)
  
  2. - new dict for <3 characters - street words (Boulevard, street, etc), conjuctions (and, or, to) / let him work on >=2? how many false positives?
     - different kinds of dict? - ex: recognize it's a street sign, using relevant dict (3RD AV)
  
  1. Dict -> evaluate by distance with weights -> for ex- 3/4 chars are 100%, do not change them in the final word
  
  3. - Transformation - If a word in a corner -> calculate the transformation between 2 frames: to check if in the new image the word is different 
     - If a word is in the corner - smaller weights for later.
     - False words than cut in corners - do we want in to autocorrect beyond the boundrybox? 
     
  5. Ask about features, specifically ResNet50 + Hourglass - can we improve (?)   
     
    
   ---  
     
   Extraction of Structured Information from Street View Imagery -
     - https://arxiv.org/pdf/1704.03549.pdf
     - https://github.com/tensorflow/models/tree/master/research/attention_ocr
     - dataset (google streetview)
   
   
   check ground truth?
   
   ---
# image annotation strong labels
- [Context data in geo-referenced digital photo collections](https://dl.acm.org/doi/pdf/10.1145/1027527.1027573)
- [NUS DATASET](https://lms.comp.nus.edu.sg/wp-content/uploads/2019/research/nuswide/NUS-WIDE.html)
