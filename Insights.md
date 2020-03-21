* [קישורים מאוזכרים במאמר 1](https://github.com/Jyouhou/SceneTextPapers)

### Difficulties
* Diversity and Variability of Text in Natural Scenes - כבר מצמצמים לשלטים?
* Complexity and Interference of Backgrounds - טיפול עצמי באמצעות מיקוד בשלטים
* Imperfect Imaging Conditions

### Methodologies
OCR:
1. pre-processing - Remove the noise, complex background, Handle the different lightning conditions
2. Detecion - create and bounding box around the text
3. Recognition

#### Detection
- Sliding window technique -  sliding window passes through the image to detect the text in that window
  - try with different window size
  - computationally expensive
  - convolutional implementation exists that can reduce the computational time
- single-shot techniques
  - YOLO
- region-based - the network proposes text region, then classify for text or not
 
##### Pipeline Simplification
2-step pipeline
anchor based default box prediction (EAST - horizontal and rotated bounding boxes)
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
long text, _Multi-Oriented Text_ (ITN - predicts affine transformation), irregular shpaes(TextSnake - the only one that was tested against multiple DBs), speed (EAST), instance segmentation, Retrieving Designated Text (DTLN - text regions, CRTR), Copmlex Background(AIF)

#### Recognition
Input: cropped text instance images: contain **one word** or **one line**
traditional steps:
1. image pre-processing - resize the cropped image to a fixed height
2. _character segmentation_ - hardest, trying to avoid by _Connectionist Temporal Classification_ or _Attention mechanism_
3. character recognition

##### CTC-based Methods
CRNN
1. pre-processing
2. 
  - convolutional layers - extract features
  - RNN: produce a character prediction for each column (label distribution for each frame)
3. transcription layer (CTC layer) - final labels (from predictions)
  - lexicon-free
  - lexicon-based - the highest probable label sequence will be predicted

FCN
1. pre-processing
2. convolutional layers
3. transcription layer (CTC layer) - final labels (from predictions)

##### Attention-based methods
- recurrent neural networks with implicitly learned characterlevel language statistics
EP - trys to estimate the probability while considering the possible occurrences of missing or superfluous characters



- Tesseract 4 for recognition - works well with straight angles

#### End-to-End
SEE - transform and crop before being fed into recognition branch

#### Aux
- syntetic Data: Model trained only on SynthText achieves state-of-the-art on many text detection datasets
- deblurring

### BENCHMARK DATASETS AND EVALUATION PROTOCOLS
Detection & Recognition
- The Street View Text (SVT) - Horizontal, not signs
- ICDAR 2013 - large and horizontal text - Detection Stats
  - Word spotting vs End-to-End evaluation
- ICDAR 2015 - google glasses, blur, small - Detection Stats
  - Word spotting vs End-to-End evaluation

Detection
- _MSRA-TD500 (2012)_ - Multi-Oriented, long - Detection Stats
  - HUST-TR400 - more training data for this dataset
 
Recognition
- IIIT 5K-Word(2012) - font, color, size and other noises, Horizontal
- SVT-Perspective (SVTP) - Google Street View, warped, not signs
- End-to-End Interpretation of the French Street Name Signs Dataset(https://arxiv.org/abs/1702.03970)
- The Street View House Numbers (SVHN)
- Scene Text dataset - korean + english - from 2nd article

#### Evaluation
- precision - the proportion of predicted text instances that can be matched to ground truth labels
- Recall - the proportion of ground truth labels that have correspondents in the predicted list
- F1-score - 2*P*R/(P+R)

Detection - DetEval, PASCAL, + modifications
Recognition+End-to-End - character-level recognition rate, word level, 


Base Nets: DenseNet, ResNet, VGG

- [ ] maybe when analyzing the sign we can get the homography (we know that it's parallel) and so we can straighten the text
---
Notes for Adir
---

**OPR Problem** - Optical Character Recognition. OCR is still a challenging problem especially when text images are taken in an unconstrained environment. For this project - **Unstructured Text**- Text at random places in a natural scene. Sparse text, no proper row structure, complex background , at random place in the image and no standard font.

Text detection techniques required to detect the text in the image and create and bounding box around the portion of the image having text. Standard objection detection techniques will also work here.

**Sliding window technique** - The bounding box can be created around the text through the sliding window technique. However, this is a computationally expensive task. In this technique, a sliding window passes through the image to detect the text in that window, like a convolutional neural network. We try with different window size to not miss the text portion with different size. **There is a convolutional implementation of the sliding window which can reduce the computational time.**

**Single Shot (YOLO) and Region based detectors**

A main distinction between text detection and general object detection is that, text are homogeneous as a whole and show locality, while general object detection are not. Thus, any part of a text instance is still text. Human do not have to see the whole text instance to know it belongs to some text.

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

2. **Decomposing into Sub-Text**

methods that only predict sub-text components, and then assemble them into a text instance, follows the standard routine of general object detection. We'll have a network which produces initial guess of the localization of possible next instance. optionally, some methods then use a refinement part to filter false positive and also correct the localization.

sub-texts level detection methods only predicts parts that are combined to make a text instance. Such sub-text mainly includes pixel-level and components-level.
On pixel-level methods, an end-to-end fully convolutional neural network learns to generate a dense prediction map, indicating whether each pixel in the original image belongs to any text instances or not. Post-processing methods, then groups pixels together (depending on which pixels belong to the same text instance). **the core oo pixel-level methods is to separate text instances from each other**

**pixellink** - learns to predict whether two adjacent pixels belong to the same text instance by adding link prediction to each pixel. - https://arxiv.org/pdf/1801.01315.pdf
It extracts text locations directly from an instance segmentation result, instead of from bounding box regression. In PixelLink, a Deep Neural Network (DNN) is trained to do two kinds of pixelwise predictions, text/non-text prediction, and link prediction. Pixels within text instances are labeled as positive (i.e., text pixels), and otherwise are labeled as negative (i.e., nontext pixels). Every pixel has 8 neighbors. For a given pixel and one of its neighbors, if they lie within the same instance, the link between them is labeled as positive, and otherwise negative.
Predicted positive pixels are joined together into Connected Components (CC) by predicted positive links

Generally speaking, sub-text level methods are more robust to the size, aspect ratio, and shape of different text instances. However, the efficiency of the postprocessing step
may depend on the actual implementation, and is slow in some cases. The lack of refinement step may also harm the performance.

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



