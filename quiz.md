
**Instructions**

This is quiz answer sheet only. In order to answer it you might need to do some code exercises provided in [quiz guideline](https://drive.google.com/open?id=16gJw-VDbFi0Tr9e5FD7zrDwSe3hZpHPT).

1.  GAN is a generative model. Please select all box that represents a             generative models 
    - [ ] Naive Bayes
    - [ ] Logistic Regression
    - [ ] Support Vector Machine
    - [ ] K-Nearest Neighbor
2.  What does adversarial refers to in GAN ?
    - [ ] Loss function
    - [ ] Optimizer
    - [ ] Input
    - [ ] Output
3.  Select all correct boxes about Generator and Discriminator
    - [ ] Generator takes input from Discriminator
    - [ ] Discriminator tries to maximize the loss function
    - [ ] Discriminator tries to generate a random noise 
    - [ ] GAN input is random vector
4.  To ensure you have correctly load the data, what is the shape of X_train and X_test respectively?
    - [ ] (60000, 28, 28, 1) and (10000, 28, 28, 1)
    - [ ] (50000, 28, 28, 1) and (5000, 28, 28, 1)
    - [ ] (60000, 32, 32, 1) and (10000, 32, 32, 1)
    - [ ] (50000, 32, 32, 1) and (5000, 32, 32, 1)
5.  While loading the Mnist dataset, in line 4, what does `X_train = (X_train.astype(np.float32) - 127.5)/127.5 ` means ?
    - [ ] Rescale the data into within the interval of -1 to 1
    - [ ] Rescale the data into within the interval of 0 to 1
    - [ ] Denormalize data
    - [ ] Reshape data
6.  If you follow the quiz guide, how many params does the generator_model has ?
    - [ ] 6,765,313
    - [ ] 6,763,777
    - [ ] 6,751,233
    - [ ] 12,544
7.  If you change the last convolution layer's filter shape into 7x7, what will be the final total parameters ?
    - [ ] 6,765,313
    - [ ] 6,763,777
    - [ ] 6,751,233
    - [ ] 12,544
8.  What does Flatten() layer means in the Discriminator?
    - [ ] To make the data standardized 
    - [ ] To make the data mean = 0
    - [ ] To reshape the data into one dimensional vector
    - [ ] To compile the previous layers
9.  What is the differences between images that are not checked by discriminator (directly generated from generator), and images that previously checked by discriminator?
    - [ ] The unchecked images are smaller in size
    - [ ] The checked images are usually better 
    - [ ] The unchecked images are usually better
    - [ ] The checked images are smaller in size
10. What do you think about losses over time while training ?
    - [ ] It will be minimized overtime
    - [ ] It will be maximized overtime
    - [ ] It will just oscilate overtime
