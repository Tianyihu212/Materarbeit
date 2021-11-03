# About the inference module
Deep learning is divided into two stages, one is the training stage and the other is the prediction stage.
Here I will introduce how I used the trained model to make predictions.
First I embed the trained model into the jina.ai framework. 
jina.ai is a very powerful retrieval framework, which can help me to achieve visualization.
I divide inference into three stages. The first is the data set preprocess. The second is to read the model and embedding of the model. 
Finally, the feature index of the data set is placed on the local computer and then the image features are input to annoy for fast retrieval. 
The figure below is the flow chart for realizing jina.ai
<br/>
![Aaron Swartz](https://github.com/Tianyihu212/Materarbeit/raw/main/inference_code/jina.ai_framwork.png)
