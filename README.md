# Highlights detector
## Idea

The model should be able to, given a soccer match, 
create an arousal function that describes the most excitement moments of the match. 
The function, combination of bells of different kurtosis and skewness, 
are a representation of the importance a video shot has inside the match.
Finally, in order to obtain the summarization video, the function should 
be filtered with adequate thresholds, as shown in the following figure. 

<img src="readme_pics\black-box.png" height="350">

## Input Data

273 goals from complete soccer matches have extracted 
from several matches of 'La liga' competition. To gather these data, a video labeler is
necessary to label and cut the necessary video highlights by hand in order to feed
a neural network able to understand its relevant features.
<img src="readme_pics\video_annotating.png" height="200">

Here (https://github.com/gioele8/video-labeler) you can find this video labeler 
to create your dataset.

## Data Preparation

## The model

The model with best results I obtained is the following:  

<img src="readme_pics\my_nn.png" height="500">

## Results

<img src="readme_pics\results_plot.png" height="400">