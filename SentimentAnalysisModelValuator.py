from keras.models import model_from_json
from keras.utils.np_utils import to_categorical
from keras.preprocessing.sequence import pad_sequences
import pickle
import pandas as pd
import sys

if len(sys.argv) < 2:
    print("Please give path to .csv Testdata as argument. Exiting...")
    exit()

data = pd.read_csv(sys.argv[1], usecols=['Topic', 'Sentiment', 'TweetText'])
#data = pd.read_csv("Test.csv", usecols=['Topic', 'Sentiment', 'TweetText'])
print("Validation Data Shape: " + str(data.shape))
print("Topics:")
print(data.Topic.value_counts())
print("Sentiments:")
print(data.Sentiment.value_counts())
data['cat'] = 0
data['sent'] = 0
data.loc[data['Topic'] == 'apple', 'cat'] = 0
data.loc[data['Topic'] == 'twitter', 'cat'] = 1
data.loc[data['Topic'] == 'google', 'cat'] = 2
data.loc[data['Topic'] == 'microsoft', 'cat'] = 3
y_cat = to_categorical(data['cat'], num_classes=4)
y_sent = to_categorical(data['sent'], num_classes=4)
if 'Topic' in data.keys():
    data.drop(['Topic'], axis=1)
if 'Sentiment' in data.keys():
    data.drop(['Sentiment'], axis=1)

with open('models/cat_tokenizer.pickle', 'rb') as handle:
    cat_tokenizer = pickle.load(handle)
with open('models/sent_tokenizer.pickle', 'rb') as handle:
    sent_tokenizer = pickle.load(handle)

max_len = 130
sequences = cat_tokenizer.texts_to_sequences(data['TweetText'].values)
x_cat = pad_sequences(sequences, maxlen=max_len)
sequences = sent_tokenizer.texts_to_sequences(data['TweetText'].values)
x_sent = pad_sequences(sequences, maxlen=max_len)

json_file = open('models/SentModel.json', 'r')
loaded_sent_model = model_from_json(json_file.read())
json_file.close()
loaded_sent_model.load_weights("models/SentModel.h5")
json_file = open('models/CatModel.json', 'r')
loaded_cat_model = model_from_json(json_file.read())
json_file.close()
loaded_cat_model.load_weights("models/CatModel.h5")
loaded_cat_model.compile(optimizer='Adagrad', loss='categorical_crossentropy', metrics=['acc'])
loaded_sent_model.compile(optimizer='Adagrad', loss='categorical_crossentropy', metrics=['acc'])

accr = loaded_cat_model.evaluate(x_cat, y_cat, verbose=0)
print('Categories: Test set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}'.format(accr[0], accr[1]))
accr = loaded_sent_model.evaluate(x_sent, y_sent, verbose=0)
print('Sentiment: Test set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}'.format(accr[0], accr[1]))
