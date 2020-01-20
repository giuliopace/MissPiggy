# MissPiggy
Image recognition deep learning model to find Miss Piggy and other pigs in an episode of the Muppet show

## Student Data
David Penz, 11703497

Giulio Pace, 11835706

## Entry point of the code

To run the code you need to execute the script ./pigfinder2000.sh as a program
the script takes the path to a video as an argument. mp4 format works for sure, other formats may cause problems sometimes.

There is one test file in the test_episodes folder available for testing.

example:
$ ./pigfinder2000.sh test_episodes/test1.mp4

## Info on architecture
### Image Classification
A very simple and shallow CNN has been implemented to create some kind of baseline evaluation for the other models. This CNN consists of the following layers:
- Input Shape 200x200x3
- 2D Convolution (Kernel Size 3x3)
- ReLU Activation
- 2D MaxPooling (Kernel Size 2x2)
- Flatten Layer as Input for FC Layers
- Dense Layer (Output 128)
- ReLU Activation
- Dense Layer (Output 1)
- Sigmoid Activation

During the research phase of this project, two architectures seemed to perform very well on image classification tasks, the VGG 19 and the Inception ResNet v2. The final version of this project is using the Inception ResNet v2 for image classification of the individual frames.
### Audio Classification
As we wanted to stick to a full Deep Learning Approach for this project, additional research was conducted to find Deep Learning models for audio classification. One of the more common approaches seems to be preprocessing the audio files into a spectrogram (using librosa) and saving the plots as image. Those images are then fed into a Convolutional Neural Network in order to classify them accordingly:
- Input Shape 64x64
- 2D Convolution (Kernel Size 3x3)
- ReLU Activation
- 2D Convolution (Kernel Size 3x3)
- ReLU Activation
- 2D MaxPooling (Kernel Size 2x2)
- Dropout
- 2D Convoluation (Kernel Size 3x3)
- ReLU Activatoin
- 2D Convolution (Kernel Size 3x3)
- ReLU Activation
- 2D MaxPooling (Kernel Size 2x2)
- Dropout
- 2D Convoluation (Kernel Size 3x3)
- ReLU Activatoin
- 2D Convolution (Kernel Size 3x3)
- ReLU Activation
- 2D MaxPooling (Kernel Size 2x2)
- Dropout
- Flatten Layer as Input for FC Layers
- Dense Layer (Output 512)
- ReLU Activation
- Dropout
- Dense Layer (Output 2)
- Softmax Activation

## Performance indicators
- F1 (recall, precision)
- ROC Curve (TPR, FPR)
- Cross Validation
- Statistical significance testing (?)
- Human Evaluation


## Timesheet of Giulio Pace
| Date        | Time    | Description                     									|
|-------------|---------|-------------------------------------------------------------------|
| 2019/10/17  | 13-16h  | attended lecture                									|
| 2019/10/18  | 09-12h  | attended lecture                									|
| 2019/11/15  | 10-12h  | brainstorming & project setup   									|
| 2019/11/16  | 16-17h  | extracted images and audio from files 							|
| 2019/11/17  | 17-21h  | started labelling images		  									|
| 2019/12/04  | 16-18h  | research state of the arts image model and started implementation | 
| 2019/12/14  | 14-17h  | implementation of image model 									| 
| 2019/12/22  | 20-22h  | research state of the art audio model								|
| 2020/01/12  | 11-15h  | completed labelling images and audio 								|
| 2019/01/14  | 16-19h  | research and implementation audio model							|
| 2019/01/17  | 14-16h  | completed audio model 											|
| 2019/01/18  | 12-16h  | Attempts at GUI, settled for a terminal script 					|
| 2019/01/19  | 10-12h  | project finished 													|




## Timesheet of David Penz
| Date        | Time    | Description                     									|
|-------------|---------|-------------------------------------------------------------------|
| 2019/10/17  | 13-16h  | attended lecture                									|
| 2019/10/18  | 09-12h  | attended lecture                									|
| 2019/11/15  | 10-12h  | brainstorming & project setup   									|
| 2019/12/04  | 16-18h  | research state of the arts image model and started implementation | 
| 2019/12/14  | 14-17h  | implementation of image model 									| 
| 2019/12/22  | 20-22h  | research state of the art audio model								|
| 2020/01/12  | 11-15h  | completed labelling images and audio 								|
| 2019/01/14  | 16-19h  | research and implementation audio model							|
| 2019/01/17  | 14-16h  | completed audio model 											|
| 2019/01/18  | 12-16h  | Attempts at GUI, settled for a terminal script 					|
| 2019/01/19  | 10-12h  | project finished 													|


## Hardware Specs
Ubuntu 18.04.3 LTS 64bit
Memory 13,6 GiB
CPU AMD® Ryzen 5 pro 3500u w/ radeon vega mobile gfx × 8 
