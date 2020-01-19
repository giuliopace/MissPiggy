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



## Info on architecture

## Hardware Specs
Ubuntu 18.04.3 LTS 64bit
Memory 13,6 GiB
CPU AMD® Ryzen 5 pro 3500u w/ radeon vega mobile gfx × 8 