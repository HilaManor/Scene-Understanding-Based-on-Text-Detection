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

#### Recognition
Input: cropped text instance images: contain **one word** or **one line**
traditional steps:
1. image pre-processing - resize the cropped image to a fixed height
2. _character segmentation_ - hardest, trying to avoid by _Connectionist Temporal Classification_ or _Attention mechanism_
3. character recognition

##### CTC-based Methods
CRNN
2. 
  - convolutional layers - extract features
  - RNN: produce a character prediction for each column (label distribution for each frame)
3. transcription layer (CTC layer) - final labels (from predictions)

FCN
2. convolutional layers
3. transcription layer (CTC layer) - final labels (from predictions)

##### Attention-based methods
- recurrent neural networks with implicitly learned characterlevel language statistics
EP - trys to estimate the probability while considering the possible occurrences of missing or superfluous characters

#### End-to-End
SEE - transform and crop before being fed into recognition branch

#### Aux
- syntetic Data: Model trained only on SynthText achieves state-of-the-art on many text detection datasets
- deblurring

### BENCHMARK DATASETS AND EVALUATION PROTOCOLS
Detection & Recognition
- The Street View Text (SVT)
- ICDAR 2013 - large and horizontal text - Detection Stats
- ICDAR 2015 - google glasses, blur, small - Detection Stats
- _MSRA-TD500 (2012)_



מה אפשר, איך, ומה אי אפשר היום

Base Nets:
DenseNet, ResNet


---
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

**In the detection process - also consists of several steps**-
1. Text blocks are extracted. 
2. The model crops and only focuses on the extracted text blocks, to extract text center line (TCL) - defined as a shrunk version of the original text line. 
Each text line represents the existance of one text instance.
3. The extractes TCL map is then split into several TCLs. Each split TCL is then concatenated (משורשר) to the original image. 
4. A semantic segmentaion model then classifies each pixel into ones that belong to the same text instance as the given TCL, and ones that do not. 

There are 3 main trends in the field of text detection

1. **Pipeline simplification** 
most recent methods have largely simplified and much shorter pipelines, which is a key to reduce error propagation and simplify the training process. The main components of these methods are end-to-end  modules (end-to-end trainable neural-network model + post-processing step that is usually much simpler than previous ones). **EAST**
EAST nakes a difference to the field of text detection with ita highly simplified pipeline and the effciency. Most famous for its speed. 

[open CV +++ "In this tutorial you will learn how to use OpenCV to detect text in natural scene images using the EAST text detector"](https://www.pyimagesearch.com/2018/08/20/opencv-text-detection-east-text-detector/)

Also - R-CNN (region convolutional neural network - for object detection) - **other method** where the second stage corrects the localization results. Rotaion Region Proposal Networks generates rotating region proposals, in orderto fit into text of arbitary orientations, instead of axis-aligned rectangles. - R2CNN, FEN

R2CNN - while most previous text detection methods are designed for detecting horizontal or near - horizonal texts, some methods try to address the arbitrary-oriented text detection problem. 



---
**Recognition**
--- 

input of these methods are cropped text instance images - contain 1 word or 1 text line.

**traditional text recognition methods*** - the task is devided into 3 steps - image pre-processing, character segmentation and character recognition. 

---

**CNN VS RNN**

**CNN** is a feed forward neural network that is generally used for Image recognition and object classification. While **RNN** works on the principle of saving the output of a layer and feeding this back to the input in order to predict the output of the layer.

CNN considers only the current input while RNN considers the current input and also the previously received inputs. It can memorize previous inputs due to its internal memory.

CNN has 4 layers namely: Convolution layer, ReLU layer, Pooling and Fully Connected Layer. Every layer has its own functionality and performs feature extractions and finds out hidden patterns.

There are 4 types of RNN namely: One to One, One to Many, Many to One and Many to Many.

RNN can handle sequential data while CNN cannot.

---

**CRNN\CTC-Based Method** - a model that stacks CNN with RNN to recognise scene text images. CRNN consists of three parts -
1. convolutional layers - which extract a feature sequence from the input image
2. Recurrent layers - predict a label distribution for each frame
3. CTC layer (transcription) - translates the per-frame predictions into the final label sequence

note - despite the progress we have seen so far, the evaluation of recognition methods falls behind the time. As most *detection* methods can detect oriented and irregular text and some even rectify them (ליישר, לסדר), the recognition of such text may seem redundant.

---
---

**End-to-End system**

In the past, text detection and recognition are usually cast as two independent sub-problems that are combined together to perform text reading from images. Recently, many end-to-end text detection and recognition systems (==text spotting systems) have been proposed.

While earlier work first detect single characters in the input image, recent systems usually detect and recognise text in word level or line level. Some of these systems first generate text proposals using a text detection model and then recognise them with another text recognition model. 


**An end-to-end scene text detection demo based on EAST and CRNN.**
 https://github.com/ppanopticon/east-crnn



