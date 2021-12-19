# Japanese-Kuzushiji-Recognition-with-Different-Neural-Network

Japanese is a  complex language with a character set consisting of complex shapes and strokes and differs vastly in the style and size based on the individual. We propose to build a program that uses neural networks to recognize the handwritten Japanese characters of the Hiragana character set from an image and convert them into digital text. Moreover, we wish to explore using language translation to provide the English meaning of the handwritten Japanese text. 

This could be particularly useful for tourists going to  Japan,  people who would like to dive into the old Japan history document, and the readers of Manga who don’t know Japanese. Manga is an increasingly popular form of content that is similar to comic books but seldom translated to other languages. Using multi-layer convolutional Neural Networks, which are known to be useful for training models on images, we will train the model to classify characters on a readily available Japanese handwritten dataset to train the model.

## DESCRIPTION OF THE DATASET:
The Kuzushiji dataset is a character database is a collection of three datasets, which are the Kuzushiji-KMNIST, Kuzushiji-49, and the Kuzushiji-kanji sets. The dataset was based on the popular MNIST dataset and follows a similar format of having 28x28 pixel grayscale images. For our project, we have decided to use the Kuzushiji-49 dataset, which consists of 270,912 images varying across 49 classes on the Hiragana Japanese character set. This dataset is also slightly imbalanced, where some classes have fewer samples than others. 

The Kuzushiji-49 dataset is further divided into 4 files, stored in the ‘.npz’ format:
Training images: Contains 232,365 samples of 28x28 pixel grayscale (8-bit) images.
Training labels: Contain 232,365 labels for the respective training image samples.
Testing images: Contains 38,547 samples of 28x28 pixel grayscale (8-bit) images.
Testing labels: Contain 38,547 labels for the respective image samples.
As they are grayscale images, there is only one channel. The dataset can be summarized in the table below.

## INPUT: 

The data are in the form of a structured file format of ‘.npz’. For training, we will need to extract the image data and label from each file to get the actual images and labels. The input to the program will be a sequence of images stored in an array, where each image is of a handwritten Japanese character. We will preprocess the image data such that it aligns with our model implementation.

## DATA PREPROCESSING:

The training images and testing images are firstly imported to NumPy arrays and reshaped to increase the rank from 3 to 4, with the extra rank responsible for the color channel for each image. Then the images are normalized through diving each data point by 255 so that each pixel will only have a range [0,1]. Before importing to the models, the images will have a rotation range of 15 degrees, and a zoom range of 0.2.

## OUTPUT: 
The output of the training will be a model which can classify handwritten Japanese characters into their text equivalent. Using this model, we will be able to classify a series of character images that form a sentence into their text counterparts. Using the final text sequence and the DeepL API for translation, the final output will be the English translation for a sequence of handwritten input image characters.

# IMPLEMENTATION: 

We first extract the labels and images from the input files to create a usable dataset on which we can train our dataset. This involved going through the dataset, retrieving the image data and its corresponding label. The training images and testing images are firstly imported to NumPy arrays and reshaped to increase the rank from 3 to 4, with the extra rank responsible for the color channel for each image. 

Then the images are normalized through diving each data point by 255 so that each pixel will only have a range [0,1]. Before importing to the models, the images will have a rotation range of 15 degrees, and a zoom range of 0.2 with ImageDataGenerator in tensorflow.preprocessing.image library.

We use the TensorFlow (abbreviated as tf below) library to implement the models. We declare each model as tf.keras.Sequential as the base model. By using the methods under tf.keras.layers, we define each model specifically using the Conv2D, MaxPooling2D, Activation, BatchNormalization, Dropout, Flatten, and Dense methods. 

The models are then compiled with an ADAM optimizer and loss function. Finally, the model will be fitted with the training data imported by the DataGenerator from the NumPy arrays. Plots are printed after each model finishes training. To measure the performance, we will have a close look at the model accuracies, loss and plot the curves. 

We then save the models so that we can use them for prediction with our own handwritten images, which the models have not seen before. Using these final models we will make predictions on the sequence of input character images, pass the prediction outputs to the DeepL API to get the English translation for the respective inputs. We then show the user the output Japanese text and its translation.

## RESULTS:

### CNN:
Accuracy and loss plots for the base CNN model:

![CNN Accuracy](/images/1.png)
![CNN Loss](/images/2.png)

### AlexNet:
Accuracy and loss plots for the AlexNet model:

![AlexNet Accuracy](/images/3.png)
![AlexNet Loss](/images/4.png)

### VGGNet:
Accuracy and loss plots for the VGGNet model:

![VGGNet Accuracy](/images/5.png)
![VGGNet Loss](/images/6.png)

### PREDICTION and DEEPL API:

Here is an example input array to the model:

![Input array to the model](/images/7.png)

An example translation from the DeepL translation API:

![Translation](/images/8.png)

## REFERENCES:
[1] Aramaki, Y., Matsui, Y., and Aizawa, K. 2016. Text detection in manga by combining connected-component-based and region-based classifications. In 2016 IEEE International Conference on Image Processing (ICIP): 2901-2905. doi.org/10.1109/ICIP.2016.7532890. 

[2] Tsai, C. 2016. Recognizing Handwritten Japanese Characters Using Deep Convolutional Neural Networks. Palo Alto, CA: Stanford University, Department of Chemical Engineering

[3] Md Zahangir Alom, Peheding Sidike , Mahmudul Hasan, Tark M. Taha, and Vijayan K. Asari. 2017. Handwritten Bangla Character Recognition Using The State-of-Art Deep Convolutional Neural Networks. https://arxiv.org/ftp/arxiv/papers/1712/1712.09872.pdf - paper which uses VGGnet and ResNet to detect Bangla characters.

[4] Mayur Bhargab Bora, Dinthisrang Daimary, Khwairakpam Amitab, Debdatta Kandar. 2019. Handwritten Character Recognition from Images using CNN-ECOC. International Conference on Computational Intelligence and Data Science (ICCIDS 2019). https://www.sciencedirect.com/science/article/pii/S1877050920307596 - paper which discusses different models which are useful for character recognition.

[5] Minh Thang Dang. 2020. Character Recognition using AlexNet. http://dangminhthang.com/computer-vision/character-recognition-using-alexnet/ - article gives a good example of using AlexNets.

[6] I Khandokar, M Hasan Md, F Ernawan, S Islam Md and M N Kabir. 2021. Handwritten character recognition using convolutional neural network. https://iopscience.iop.org/article/10.1088/1742-6596/1918/4/042152

[7] M. Jain, G. Kaur, M. P. Quamar and H. Gupta, "Handwritten Digit Recognition Using CNN," 2021 International Conference on Innovative Practices in Technology and Management (ICIPTM), 2021, pp. 211-215, doi: 10.1109/ICIPTM52218.2021.9388351.


[8] L. Chen, S. Wang, W. Fan, J. Sun and S. Naoi, "Beyond human recognition: A CNN-based framework for handwritten character recognition," 2015 3rd IAPR Asian Conference on Pattern Recognition (ACPR), 2015, pp. 695-699, doi: 10.1109/ACPR.2015.7486592.

[9] Mudhsh, M.; Almodfer, R. Arabic Handwritten Alphanumeric Character Recognition Using Very Deep Neural Network. Information 2017, 8, 105. https://doi.org/10.3390/info8030105

[10] N. S. Rani, A. C. Subramani, A. Kumar P. and B. R. Pushpa, "Deep Learning Network Architecture based Kannada Handwritten Character Recognition," 2020 Second International Conference on Inventive Research in Computing Applications (ICIRCA), 2020, pp. 213-220, doi: 10.1109/ICIRCA48905.2020.9183160.

[11] Lee S-G, Sung Y, Kim Y-G, Cha E-Y. Variations of AlexNet and GoogLeNet to Improve Korean Character Recognition Performance. Journal of Information Processing Systems [Internet]. 2018 Feb 28;14(1):205–17. Available from: https://doi.org/10.3745/JIPS.04.0061
