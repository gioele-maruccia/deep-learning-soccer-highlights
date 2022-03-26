# Highlights detector
## Idea

The model should be able to, given a soccer match, 
create an arousal function that describes the most excitement moments of the match. 
The function, combination of bells of different kurtosis and skewness, 
are a representation of the importance a video shot has inside the match.
Finally, in order to obtain the summarization video, the function should 
be filtered with adequate thresholds, as shown in the following figure. 

<img src="readme_pics\black-box.png" height="500">


## The model

The model with best results I obtained is the following:  

<img src="readme_pics\my_nn.png" height="500">

## Results

<img src="readme_pics\results_plot.png" height="500">