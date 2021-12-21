# Human Detection using HOG

In my final project of CS231 class. I'm working on "Human Detection" using HOG feature in order to find out basic model like "Dalal & Triggs Model" and "Deformable Part Models".
Through this final project, I've known some basic idea about:
* HOG feature
* Sliding Window
* Image Pyramid
* Concept about "Feature Descriptor"
* Filter
* Cross-Correlation
* Idea of part-based-model 
<br>
Although the powerfull of modern Deep Learning Model may surpass these old model, but after this project I've learned lots of knowledge about Computer Vision area.

# HOG feature
HOG feature using information about gradient to create a histogram in each cell (8x8 pixel), then normalize histogram and concatentate those vector to create the final feature descriptor. 

# Dalal & Triggs Model
  Positive sample is image patch which have people in that and many Negative samples which is background image patch don't have people there. Extract HOG feature, training using SVM to creat a classifier. Using Sliding Window approach, a window which size is 64x128 slide through all region of the image. At each region, extract HOG feature and then apply filter to predict label. Filter is an array of weight got affter training model. Cons of Dalal & Triggs Model is it can not detect human in various poses, different viewpoint.
 
 # Deformable Part Models
  Not only detect a human with one filter (called "root filter" which detect whole human) but also detect human with a set of "part filter". Score is sum of root filter score, part filters score and minute deformation cost. Deformation cost of each part is penaly of part relative to root filter. Obtaining root filter, part filter, deformation cost by training Latent SVM.
  
 # Code
  Train DPM from the begining, visit here "https://github.com/rbgirshick/voc-dpm" to find the original paper and code using MatLab. OpenCV also have dpm function in opencv-contrib for detecting human, airplane, dog, etc.
  To run the code, use CMake to build opencv and opencv-contrib.
