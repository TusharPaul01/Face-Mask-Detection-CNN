# Face-Mask-Detection-CNN
Using Convolutional Neural Network model

Face mask detection has become crucial in ensuring public safety during the COVID-19 pandemic. With the help of deep learning techniques, we can develop a model that can accurately determine whether a person is wearing a face mask or not. In this project, we utilize a CNN architecture, which excels at image classification tasks.

To begin, we import the necessary libraries such as TensorFlow, Keras, cv2 (OpenCV), and Matplotlib. These libraries provide the tools we need for image preprocessing, model building, and visualization.

We then load and preprocess the dataset. We have a dataset of approx. 10,000 images, with a total size of around 330 MB, we split it into training and validation sets. The dataset is loaded using the image_dataset_from_directory function, which conveniently handles the directory structure and class labels. We also normalize the pixel values between 0 and 1 to enhance model performance.
Dataset used : https://www.kaggle.com/datasets/ashishjangra27/face-mask-12k-images-dataset

Next, we build the CNN model. The model consists of multiple convolutional and max-pooling layers, which capture and extract relevant features from the input images. We use the ReLU activation function for the convolutional layers to introduce non-linearity. The final layer of the model will provide the classification output.

With the model architecture defined, we can now train the model on the training dataset. We specify the number of epochs, batch size, and other training parameters to optimize the model's performance. During training, the model learns to differentiate between masked and unmasked faces by adjusting the weights of the network's parameters.

After training, we can evaluate the model's performance using the validation dataset. 
#### Accuracy attained 98.8%

To put the model into practical use, we can upload an image for inference. Using the trained model, we can apply it to the input image and determine whether the person in the image is wearing a mask or not. The output can be displayed along with the image using Matplotlib for easy visualization.
