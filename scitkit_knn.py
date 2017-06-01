from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from scipy import stats
from scipy import spatial
from sys import argv
import argparse

import numpy  as np

def separateFiles(trainningFile, testFile, sections):	
	if(len(sections) == 0):
		with open(trainningFile, 'r') as file:
			labels = []
			characteristics = []

			data = np.loadtxt(file)
			for line in data:
				labels.append(line[len(data[0])-1])		

			for line in data:
				characteristics.append(line[:-1])

			trainningLabelsFile = trainningFile+'_labels.txt'
			trainningCharacteristicsFile = trainningFile+'_characteristics.txt'

			np.savetxt(trainningLabelsFile, labels)
			np.savetxt(trainningCharacteristicsFile, characteristics)

		with open(testFile, 'r') as file:
			labels = []
			characteristics = []
					
			data = np.loadtxt(file)
			for line in data:
				labels.append(line[len(data[0])-1])		

			for line in data:
				characteristics.append(line[:-1])

			testLabelsFile = testFile+'_labels.txt'
			testCharacteristicsFile = testFile+'_characteristics.txt'

			np.savetxt(testLabelsFile, labels)
			np.savetxt(testCharacteristicsFile, characteristics)
	
	else:
		with open(trainningFile, 'r') as file:
			labels = []
			characteristics = []

			data = np.loadtxt(file)
			for s in sections:
				for line in data:
					if(line[len(data[0])-1] == s):
						labels.append(line[len(data[0])-1])		

			for s in sections:
				for line in data:
					if(line[len(data[0])-1] == s):
						characteristics.append(line[:-1])

			trainningLabelsFile = trainningFile+'_labels.txt'
			trainningCharacteristicsFile = trainningFile+'_characteristics.txt'

			np.savetxt(trainningLabelsFile, labels)
			np.savetxt(trainningCharacteristicsFile, characteristics)

		with open(testFile, 'r') as file:
			labels = []
			characteristics = []
					
			data = np.loadtxt(file)
			for s in sections:
				for line in data:
					if(line[len(data[0])-1] == s):
						labels.append(line[len(data[0])-1])		


			for s in sections:
				for line in data:
					if(line[len(data[0])-1] == s):
						characteristics.append(line[:-1])

			testLabelsFile = testFile+'_labels.txt'
			testCharacteristicsFile = testFile+'_characteristics.txt'

			np.savetxt(testLabelsFile, labels)
			np.savetxt(testCharacteristicsFile, characteristics)


def extract(fileName):
	with open(fileName) as file:
		data = np.loadtxt(file)
	return data

def buildConfusionMatrix(classified, test_labels):	
	confusionMatrix = confusion_matrix(test_labels, classified)
	return confusionMatrix



if __name__ == '__main__':
	arg_parser = argparse.ArgumentParser()
	arg_parser.add_argument("trainning_file", help="arquivo de treinamento")
	arg_parser.add_argument("test_file", help="arquivo de teste")
	arg_parser.add_argument("k", help="k")
	arg_parser.add_argument('--sections', nargs='+', type=float, default=[], help='List of sections to use.')
	arg_parser = arg_parser.parse_args()
	separateFiles(arg_parser.trainning_file, arg_parser.test_file, sorted(arg_parser.sections))
	
	trainningLabelsFile = arg_parser.trainning_file+'_labels.txt'
	trainningCharacteristicsFile = arg_parser.trainning_file+'_characteristics.txt'	

	trainning_labels = extract(trainningLabelsFile)
	trainning_characteristics = extract(trainningCharacteristicsFile)

	testLabelsFile = arg_parser.test_file+'_labels.txt'
	testCharacteristicsFile = arg_parser.test_file+'_characteristics.txt'

	test_labels = extract(testLabelsFile)
	test_characteristics = extract(testCharacteristicsFile)



	neigh = KNeighborsClassifier(n_neighbors=arg_parser.k)
	neigh.fit(trainning_characteristics, trainning_labels)

	classified = neigh.predict(test_characteristics)
	prob = neigh.predict_proba(test_characteristics)


	matrix = buildConfusionMatrix(classified, test_labels)

	for i in matrix:
		print i

	accuracy = accuracy_score(test_labels, classified)
	
	print accuracy