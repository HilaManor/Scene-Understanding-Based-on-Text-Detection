# רשמים

## Scene Text Detecion and Recognition: The Deep Learning Era
* [קישורים מאוזכרים במאמר](https://github.com/Jyouhou/SceneTextPapers)

Spotting text is composed of 2 steps: **Detection** and then **Recognition**

### Difficulties
* Diversity and Variability of Text in Natural Scenes - כבר מצמצמים לשלטים?
* Complexity and Interference of Backgrounds - טיפול עצמי באמצעות מיקוד בשלטים
* Imperfect Imaging Conditions

### Methodologies
#### Detection
##### Pipeline Simplification
2-step pipeline
anchor based default box prediction (EAST)
second stage corrects localization results based on features obtained by ROI pooling (R-CNN, R2CNN - different sizes)

##### Decomposing into Sub-Text
every text is composed of more text. humans can see one letter and know that it belongs to text
- Text instance level methods: may suffer from lack of end-to-end optimization.
  - network for region-proposal
- sub-text level methods: more robust to the size, aspect ratio, and shape of different text
instances. However, efficiency of the postprocessing step is slow in some cases
  - pixel-level: is each pixel belongs to text? then post-processing to group them smartly (PixelLink)
  - components-lvel: text segments such as one or more characters(SegLink, Corner localization-> *multi-oriented text*)
##### Specific Targets
long text, _Multi-Oriented Text_ (ITN - predicts affine transformation), irregular shpaes(TextSnake), speed (EAST), instance segmentation, Retrieving Designated Text (DTLN - text regions, CRTR), Copmlex Background(AIF)


מה אפשר, איך, ומה אי אפשר היום
Base Nets:
DenseNet, ResNet

---
Notes for Adir
---


Classify existing methods - 
1. text detection that detects and localizes the existance of text in natural image.
2. recogntion system that transcribes and converts the content of the detected text region into symbols.
3. end-to-end system that performs both text detection and recognition in one single pipeline.
4. auxiliary methods that aim to support.

**Detection**

The detection of scene text has a different set of characteristics and challenges that require unique methologies and solutions. 

**In the setection process - also consists of several steps**-
1. Text blocks are extracted. 
2. The model crops and only focuses on the extracted text blocks, to extract text center line (TCL) - defined as a shrunk version of the original text line. 
Each text line represents the existance of one text instance.
3. The extractes TCL map is then split into several TCLs. Each split TCL is then concatenated (משורשר) to the original image. 
4. A semantic segmentaion model then classifies each pixel into ones that belong to the same text instance as the given TCL, and ones that do not. 

There are 3 main trends in the field of text detection

1. **Pipeline simplification** 
most recent methods have largely simplified and much shorter pipelines, which is a key to reduce error propagation and simplify the training process. The main components of these methods are end-to-end  modules (end-to-end trainable neural-network model + post-processing step that is usually much simpler than previous ones). **EAST**
EAST nakes a difference to the field of text detection with ita highly simplified pipeline and the effciency. Most famous for its speed. 

https://www.pyimagesearch.com/2018/08/20/opencv-text-detection-east-text-detector/ - open CV ((Open Source Computer Vision Library)) +++ "In this tutorial you will learn how to use OpenCV to detect text in natural scene images using the EAST text detector".

Also - R-CNN (region convolutional neural network - for object detection) - **other method** where the second stage corrects the localization results. Rotaion Region Proposal Networks generates rotating region proposals, in orderto fit into text of arbitary orientations, instead of axis-aligned rectangles. - R2CNN, FEN

R2CNN - while most previous text detection methods are designed for detecting horizontal or near - horizonal texts, some methods try to address the arbitrary-oriented text detection problem. 





