# SentimentAnalysis

Models are created in the Multiclass Jupyther Notebook and exported to the models folder. For testing the Validator Script can be used. It takes a path to a CSV Dataset as argument. Dataset structure has to be the same as the original Test and Trainset. Models folder has to be at the same location as the validator script.
Two models are categorising the tweets into the topics Google, Microsoft, Twitter and Apple, also into the sentiments of positive, negative, neutral and irrelevant.
Script showes the accuracy of the models on the testset given in the argument.

Keras and Pandas are needed.
Validator tested with Python 3.6
