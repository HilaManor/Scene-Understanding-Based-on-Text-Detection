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
long text, _Multi-Oriented Text_ (ITN - predicts affine transformation), irregular shpaes(TextSnake), speed (EAST), instance segmentation(


מה אפשר, איך, ומה אי אפשר היום
Base Nets:
DenseNet, ResNet
