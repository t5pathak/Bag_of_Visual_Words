# Bag of Visual Words - BOVW

Bag of visual words (BOVW) is commonly used in image classification. Its concept is adapted from information retrieval and NLP’s bag of words (BOW). In bag of words (BOW), we count the number of each word appears in a document, use the frequency of each word to know the keywords of the document, and make a frequency histogram from it. We treat a document as a bag of words (BOW). We have the same concept in bag of visual words (BOVW), but instead of words, we use image features as the “words”. Image features are unique pattern that we can find in an image. <br /> <br />
The general idea of bag of visual words (BOVW) is to represent an image as a set of features. Features consists of keypoints and descriptors. Keypoints are the “stand out” points in an image, so no matter the image is rotated, shrink, or expand, its keypoints will always be the same. And descriptor is the description of the keypoint. We use the keypoints and descriptors to construct vocabularies and represent each image as a frequency histogram of features that are in the image. From the frequency histogram, later, we can find another similar images or predict the category of the image. <br />

[Introduction Credits - Towards Data Science](https://towardsdatascience.com/bag-of-visual-words-in-a-nutshell-9ceea97ce0fb)

# Code - What has been implemented
1. Built a vocabulary of visual words by sampling local features from the training set and then clustering them with kmeans.
2. Computed dense-SIFT features on the images.
3. Represented the training and testing images as histograms of visual words by counting how many SIFT descriptors fall into each cluster in our visual word vocabulary.
4. Normalized the histogram so that image size does not dramatically change the bag of feature magnitude
5. Performed classification by training one-vs-all linear SVMs using sklearn
6. Ploted the confusion matrix without and with normalization to estimate the performance of the model.

# Results - Accuracy Scores

| No. of Images |Accuracy Score| Normalization |
|-------------|-------------|-------------|
|50|41.25%|NO|
|70|47.5%|NO|
|50|50.00%|YES|
|70|52.5%|YES|

With an increase in number of visual words, the accuracy increases. The accuracy is better (for the same number of visual words) in the case of "with normalisation"

# Directory Structure
- ```src``` folder contains the source code. 
- ```SVM``` contains all the .pkl files.
- ```dataset``` contains all a simplified version of SUN dataset.
 
# Running the code
1.Run the code while maintaining the directory structure.
