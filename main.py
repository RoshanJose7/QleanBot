import nltk
import numpy
import tflearn
import tensorflow
import random
import json
import pickle

from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

training = []
output = []

# Read data from JSON file
with open("intents.json") as file:
	data = json.load(file)

# Check if data already exists
try:
    with open("data.pickle", "rb") as f:
        words, labels, training, output = pickle.load(f)

except:
	# If data does not exist pre process it again
	# Global Variables
	words = []
	labels = []
	docs_x = []
	docs_y = []

	for intent in data["intents"]:
		for pattern in intent["patterns"]:
			tag = intent["tag"]
			# Stemming: getting the root of each word: example: - What's -> what
			# Tokenize: split sentence into words
			wrds = nltk.word_tokenize(pattern)
			words.extend(wrds)

			docs_x.append(wrds)
			docs_y.append(tag)

			if tag not in labels:
				labels.append(tag)

	# Remove duplicate words and put in sorted order
	words = [stemmer.stem(w.lower()) for w in words if w != "?"]
	words = sorted(list(set(words)))

	labels = sorted(labels)

	# Pre-processing data
	# Start training and testing output
	# 1 hot encoded array
	training = []
	output = []

	out_empty = [0 for _ in range(len(labels))]

	for x, doc in enumerate(docs_x):
		bag = []

		wrds = [stemmer.stem(w) for w in doc]

		for w in words:
			if w in wrds:
				bag.append(1)
			else:
				bag.append(0)

		output_row = out_empty[:]
		output_row[labels.index(docs_y[x])] = 1

		# add to training and output array
		training.append(bag)
		output.append(output_row)

	training = numpy.array(training)
	output = numpy.array(output)

	with open("data.pickle", "wb") as f:
		pickle.dump((words, labels, training, output), f)


# ML Model
tensorflow.compat.v1.reset_default_graph()

net = tflearn.input_data(shape=[None, len(training[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
net = tflearn.regression(net)

model = tflearn.DNN(net)

try:
	# Check if model already trained and saved
	open("model.tflearn.index", 'r')
	model.load("model.tflearn")

except:
	# If not train again
	model.fit(training, output, n_epoch=1000, batch_size=8, show_metric=True)
	model.save("model.tflearn")


# Process user input and check if word in sentence
def bag_of_words(s, words):
	bag = [0 for _ in range(len(words))]

	s_words = nltk.word_tokenize(s)
	s_words = [stemmer.stem(word.lower()) for word in s_words]

	for se in s_words:
		for i, w in enumerate(words):
			if w == se:
				bag[i] = 1

	return numpy.array(bag)


def chat():
	print("Start talking with the bot!")
	print("Type 'quit' to stop!")

	while True:
		inp = input("You: ")

		if inp.lower() == "quit":
			break

		# Run Prediction
		results = model.predict([bag_of_words(inp, words)])[0]
		results_index = numpy.argmax(results)
		tag = labels[results_index]

		# Get random response from responses array
		if results[results_index] > 0.6:
			# Print response only if accuracy greater than 70%
			for tg in data["intents"]:
				if tg["tag"] == tag:
					responses = tg['responses']
			
			print("Jarves: " + random.choice(responses))

		else:
			print("Sorry, I didn't understand that!")

chat()
