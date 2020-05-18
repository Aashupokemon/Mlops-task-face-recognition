# Mlops-task-face-recognition
LinuxWorld India

# TASK FOR FACE RECOGNITION under the MLOps Training under the mentorship of Mr.  Vimal Daga Sir.

I want to thanks Vimal Daga Sir for giving us such an amazing task and also Thanks to Arsh Mishra for continuous help during this task.

In this task I used the #VGG16 pre trained model and then using the concept of #transferlearning I recreated the model and used for

predicting my own face.

In this task it took About 30 mins for training model with 5 epochs and then saving the model.

After saving load the model and predict the image using the model.

The model predicted the right image as the #VGG16  model is highly efficient model.

# Introduction

VGG16 is a convolution neural net (CNN ) architecture which was used to win ILSVR(Imagenet) competition in 2014. It is considered to be one of the excellent vision model architecture till date. Most unique thing about VGG16 is that instead of having a large number of hyper-parameter they focused on having convolution layers of 3x3 filter with a stride 1 and always used same padding and maxpool layer of 2x2 filter of stride 2. It follows this arrangement of convolution and max pool layers consistently throughout the whole architecture. In the end it has 2 FC(fully connected layers) followed by a softmax for output. The 16 in VGG16 refers to it has 16 layers that have weights. This network is a pretty large network and it has about 138 million (approx) parameters.

Face Recognition system using the concept of Transfer Learning

Transfer learning make use of the knowledge gained while solving one problem and applying it to a different but related problem.

# Requirement of transfer learning

When we train the network on a large dataset(for example: ImageNet) , we train all the parameters of the neural network and therefore the model is learned. It may take hours on your GPU.

so to reduce the time we do is that we freez all the layers of the pre trained model and after that we add a new layer so that it can be trained and then train it to detect the new object.

# Steps for creating model

1: I am using VGG16 as the pretrained model and will further do the modifications on it to build the face recognition system.

2: I created a dataset and splitted it into parts as training data and the testing data . once the model will be trained then i will use my test dataset to see wether the model is predicting accurately or not

3: Here I first importing all the libraries which i will need to implement VGG16. I will be using Sequential method as I am creating a
sequential model. Sequential model means that all the layers of the model will be arranged in sequence. Here I have imported ImageDataGenerator from keras.preprocessing. The objective of ImageDataGenerator is to import data with labels easily into the model. It is a very useful class as it has many function to rescale, rotate, zoom, flip etc. The most useful thing about this class is that it doesn’t affect the data stored on the disk. This class alters the data on the go while passing it to the model.

4: Here I will be using Adam optimiser to reach to the global minima while training out model. If I am stuck in local minima while training then the adam optimiser will help us to get out of local minima and reach global minima. We will also specify the learning rate of the optimiser, here in this case it is set at 0.001. If our training is bouncing a lot on epochs then we need to decrease the learning rate so that we can reach global minima.

5: I can check the summary of the model which I created by using the code below.

    # model.summary()

6: I am using model.fit_generator as I am using ImageDataGenerator to pass data to the model. I will pass train and test data to fit_generator. In fit_generator steps_per_epoch will set the batch size to pass training data to the model and validation_steps will do the same for test data. You can tweak it based on your system specifications.

7: After executing the above line the model will start to train and you will start to see the training/validation accuracy and loss.

Finally the image is predicted right.

