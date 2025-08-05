# Machine Learning Problem - Adversarial Noise
Machine Learning Programming Challenge - Journal

Assumption #1: We have an Image Classification model, it can be any model (e.g. VGG16). 

The goal of the challenge is to create a function/API/model/system that adds noise to an image to modify the output of the Image Classification model. 

Inputs:
1. Input -> Image belonging to one of the labels the Img Classification model is trained for. Assumption #2 we can use the images from the validation/testing set.
2. Input -> String label that the input image should be classified to. 

Outputs:
1. Modified Image that if passed to our Image Classification model, it will be classified with the given input string label.

The constraints are the following:
1. The model cannot be modified, but can be used to extract feature embedding vectors.
2. The modified output image should look the same as the input image to the human eye.
3. The possible labels are limited to the labels that the image classification model was trained on.

Discussion/Ideas
Image classification models have a series of convolution blocks that extract features from an input image.
The features extracted by the convolutions end up becoming a vector of features.
The model will then have a classifier head that will classify this features vector to one of the known class labels.

So essentially, we want to make that classifier head to output the label we want. 
That means that we need to modify the feature vector of the input image to look similar to a feature vector of the label we choose.
To do that, we need to know what is a "typical feature vector" for the different class labels.

So the pipeline is looking something like...
[input image] --> [extract feature vector; "input vector"] 
[input desired label] --> [find typical feature vector for that label; "target vector"]

our_function(input_vector, target_vector):
    // modify input vector to look similar to target vector
    return modified_vector

Now, if [modified_vector] --> [black box img classification model] then we expect [black box img classification model] --> [input desired label]

However, we are still missing one more step...
We need to generate from the modified_vector a similar image that looks similar to the input_image
For this can we create a GAN? to create an image from the modified_vector

How are we evaluating this?
1. First we need to evaluate if our modified_vector is classified as the desired_label.
2. We need to evaluate how different is the image generated from our modified_vector to our input image


Initial thoughts
1. We need to extract "typical feature vectors" for all possible labels.
    1.1 What if we just encode several image of each class using the same img classification model
    1.2 We then cluster these feature vectors
    1.3 Our centroid is the "typical feature vector"
2. To add noise to our input image
    2.1 Extract the feature vector using the image classification model
    2.2 Get the known "typical feature vector" for the given desired label
    2.3 Make the input image vector to be similar to the typical feature vector,
            move the vector in space to be closer in distance to the typical vector?
            not sure how to do this, will need to research on how features vector can be modified
3. Generate an output image from the modified vector
    3.1 my first thought is to train a GAN to do this but not sure how this will affect the prediction of the image classification model.
    3.2 I'll also need to research this, to evaluate how viable is a GAN for this problem
4. Create a full pipeline and deploy it as an API, to be available for anyone that wants to use it.
    4.1 This should be easy, we can use FastAPI library to quickly create a function that receives the input image and label
    4.2 Do all the processing, create the image and return it.


Update #1
After some research I don't think I need a GAN. We can reconstruct the image using an optimisation loop with gradient descent.
Look at: 
1. https://github.com/mndu/guided-feature-inversion
2. https://github.com/guanhuaw/MIRTorch
3. https://github.com/KamitaniLab/icnn 

Need to start the implementation. <br>
This is my plan...<br>
- [X] Predict function. Using pytorch create a predict function that uses resnet50, vgg16, or vgg19 to make image classification predictions.
- [X] Extract Features function (Encoder). Using the same model from the predict function, create a function that doesn't do prediction but returns vector of extracted features.
- [X] Reconstruct Image. Create the function that given a vector of features we reconstruct an image that is similar to a given input image. This is going to be our reconstruction function that will generate the output.
- [ ] Define list of possible labels for the user to select from. Create a small sample dataset for each label. 
- [ ] Extract features vectors for each label. Cluster them. Get the centroid and save it as our "typical features vector" for each label.
- [ ] Modify our reconstruct image function to use the extracted centroids of the labels instead of random noise. 
- [ ] Test full pipeline:
    1. Select input image.
    2. Select desired expected label.
    3. Encode input image.
    4. Get "typical features vector" for desired label.
    5. Add noise to encoded input image.
    6. Reconstruct modified encoded image.
    7. Save reconstruct image as output.jpg
