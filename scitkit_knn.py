from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from scipy import stats
from scipy import spatial
from sys import argv
import numpy  as np

def separateFiles(trainningFile, testFile):
	with open(trainningFile, 'r') as file:
		data = np.loadtxt(file)
		labels = data[:, [len(data[0])-1]]
		characteristics = np.delete(data, -1, 1)

		trainningLabelsFile = trainningFile+'_labels.txt'
		trainningCharacteristicsFile = trainningFile+'_characteristics.txt'

		np.savetxt(trainningLabelsFile, labels)
		np.savetxt(trainningCharacteristicsFile, characteristics)

	with open(testFile, 'r') as file:
		data = np.loadtxt(file)
		labels = data[:, [len(data[0])-1]]
		characteristics = np.delete(data, -1, 1)

		testLabelsFile = testFile+'_labels.txt'
		testCharacteristicsFile = testFile+'_characteristics.txt'

		np.savetxt(testLabelsFile, labels)
		np.savetxt(testCharacteristicsFile, characteristics)

def extract(fileName):
	with open(fileName) as file:
		data = np.loadtxt(file)
	return data


def buildConfusionMatrix(classified, trainning_labels, test_labels):
	n_rows = len(trainning_labels)
	n_columns = len(trainning_labels)
	
	# Creates a list with unique labels names from the trainning labels
	used = set()
	distinctLabels = [x for x in trainning_labels if x not in used and (used.add(x) or True)]
	confusionMatrix = confusion_matrix(test_labels, classified)

	return confusionMatrix



if __name__ == '__main__':
	separateFiles(argv[1], argv[2])


	trainningLabelsFile = argv[1]+'_labels.txt'
	trainningCharacteristicsFile = argv[1]+'_characteristics.txt'	

	trainning_labels = extract(trainningLabelsFile)
	trainning_characteristics = extract(trainningCharacteristicsFile)

	testLabelsFile = argv[2]+'_labels.txt'
	testCharacteristicsFile = argv[2]+'_characteristics.txt'

	test_labels = extract(testLabelsFile)
	test_characteristics = extract(testCharacteristicsFile)

	neigh = KNeighborsClassifier(n_neighbors=1)
	neigh.fit(trainning_characteristics, trainning_labels)

	classified = neigh.predict(test_characteristics)
	prob = neigh.predict_proba(test_characteristics)


	matrix = buildConfusionMatrix(classified, trainning_labels, test_labels)

	for i in matrix:
		print i

	accuracy = accuracy_score(test_labels, classified)
	
	print accuracy