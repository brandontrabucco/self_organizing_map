/*
 * NeuronMap.cpp
 *
 *  Created on: Jul 14, 2016
 *      Author: trabucco
 */

#include "NeuronMap.h"

NeuronMap::NeuronMap(int connectionsPerNeuron, int *_dim, unsigned int _type, double _width, double _learningRate, double _decayRate) {
	// TODO Auto-generated constructor stub

	type = _type;
	width = _width;
	learningRate = _learningRate;
	decayRate = _decayRate;
	dim = _dim;

	int nNeurons = 1;
	for (int i = 0; i < type; i++) {
		nNeurons *= dim[i];
	} for (int i = 0; i < nNeurons; i++) {
		map.push_back(Neuron());
		vector<Synapse> temp;
		for (int j = 0; j < connectionsPerNeuron; j++) {
			temp.push_back(Synapse());
		} connections.push_back(temp);
	}
}

NeuronMap::~NeuronMap() {
	// TODO Auto-generated destructor stub
}


/**
 *
 * 	A recursive loop to simulate an n-dimensional vector space
 *
 */
void NeuronMap::activateNeuron(vector<double> input, int dimension, int index) {
	int idx;
	if (dimension < type) {
		for (int i = 0; i < dim[dimension]; i++) {
			idx = index * dim[dimension] + i;
			activateNeuron(input, dimension + 1, idx);
			if (dimension == (type - 1)) {
				double sum = 0;
				for (int l = 0; l < connections[idx].size(); l++) sum += connections[idx][l].get(input[l]);
				map[idx].get(sum);
			}
		}
	} else return;
}


/**
 *
 * 	A recursive loop to simulate an n-dimensional vector space
 *
 */
void NeuronMap::findLargestDistance(vector<double> input, double &bestDistance, int &bestIndex, int dimension, int index) {
	int idx;
	if (dimension < type) {
		for (int i = 0; i < dim[dimension]; i++) {
			idx = index * dim[dimension] + i;
			findLargestDistance(input, bestDistance, bestIndex, dimension + 1, idx);
			if (dimension == (type - 1)) {
				double temp = distance(connections[idx], input);
				if (temp < bestDistance) {
					bestDistance = temp;
					bestIndex = idx;
				}
			}
		}
	} else return;
}


/**
 *
 * 	A recursive loop to simulate an n-dimensional vector space
 *
 */
void NeuronMap::getCorrection(vector<double> input, int bestIndex, int dimension, int index, bool print) {
	int idx;
	if (dimension < type) {
		for (int i = 0; i < dim[dimension]; i++) {
			idx = index * dim[dimension] + i;
			getCorrection(input, bestIndex, dimension + 1, idx, print);
			if (dimension == (type - 1)) {
				for (int l = 0; l < connections[idx].size(); l++) {
					connections[idx][l].correction += learningRate * neighborhood(bestIndex, idx) * (input[l] - connections[idx][l].weight);
				}
			}
		}
	} else return;
}


/**
 *
 * 	A recursive loop to simulate an n-dimensional vector space
 *
 */
void NeuronMap::updateWeight(int dimension, int index) {
	int idx;
	if (dimension < type) {
		for (int i = 0; i < dim[dimension]; i++) {
			idx = index * dim[dimension] + i;
			updateWeight(dimension + 1, idx);
			if (dimension == (type - 1)) {
				for (int l = 0; l < connections[idx].size(); l++) {
					connections[idx][l].weight += connections[idx][l].correction;
					connections[idx][l].correction = 0;
				}
			}
		}
	} else return;
}


/**
 *
 * 	Calculate correction
 * 	Update weights every cycle
 *
 */
void NeuronMap::online(vector<double> input, bool print) {
	// find the winning node
	int bestIndex = 0;
	double bestDistance = distance(connections[0], input);
	findLargestDistance(input, bestDistance, bestIndex, 0, 0);
	// update the weights of all nodes in the map
	getCorrection(input, bestIndex, 0, 0, print);
	updateWeight(0, 0);
	width *= decayRate;
	learningRate *= decayRate;
}


/**
 *
 * 	Calculate and sum correction
 * 	Update weights every n cycles
 *
 */
void NeuronMap::batch(vector<double> input, bool print, bool update) {
	// find the winning node
	int bestIndex = 0;
	double bestDistance = distance(connections[0], input);
	findLargestDistance(input, bestDistance, bestIndex, 0, 0);
	// update the weights of all nodes in the map
	getCorrection(input, bestIndex, 0, 0, print);
	if (update) {
		updateWeight(0, 0);
		width *= decayRate;
		learningRate *= decayRate;
	}
}


/**
 *
 * 	Pass an input vector through the map
 *
 */
vector<double> NeuronMap::recognize(vector<double> input) {
	activateNeuron(input, 0, 0);
	vector<double> output;
	for (int i = 0; i < map.size(); i++) {
		output.push_back(map[i].activation);
	} return output;
}


/**
 *
 * 	Get Euclidean Distance between the input and weight vectors
 *
 */
double NeuronMap::distance(vector<Synapse> s, vector<double> t) {
	double sum = 0;
	if (s.size() == t.size()) {
		for (int i = 0; i < s.size(); i++) {
			sum += (s[i].weight - t[i]) * (s[i].weight - t[i]);
		} return sqrt(sum);
	} else return 0;
}


/**
 *
 * 	Weight Neurons with a Euclidean Distance
 * 	Neurons closer to winner in n-dim space have higher weight
 *
 */
double NeuronMap::neighborhood(int a, int b) {
	double positionA[type], positionB[type];
	for (int i = 0; i < type; i++) {
		// get the linear ID of the element
		positionA[i] = (double)a;
		positionB[i] = (double)b;

		// calculate the dimensional modulation
		int modulo = 1;
		for (int j = 0; j <= i; j++)
			modulo *= dim[j];

		// modulate the ID to repeat per dimension
		positionA[i] = (double)(((int)positionA[i]) % modulo);
		positionB[i] = (double)(((int)positionB[i]) % modulo);

		// calculate the dimension index
		for (int j = 0; j < i; j++)
			positionA[i] /= (double)dim[j];
		positionA[i] = floor(positionA[i]);
	} double sum = 0;
	for (int i = 0; i < type; i++) sum += pow((positionA[i] - positionB[i]), 2);
	sum = sqrt(sum);
	return exp((-1*(sum * sum)) / (2 * width));
}

struct tm *currentDate() {
	time_t t = time(NULL);
	struct tm *timeObject = localtime(&t);
	return timeObject;
}


/**
 *
 * 	Save the current instance of the network
 * 	Output a file of weights and biases
 *
 */
void NeuronMap::toFile(int iteration, int trainingSet, int epoch) {
	ostringstream fileName;
	fileName << "/stash/tlab/trabucco/ANN_Saves/" <<
			(currentDate()->tm_year + 1900) << "-" << (currentDate()->tm_mon + 1) << "-" << currentDate()->tm_mday <<
			"_Single-Core-SOM-Save-" << iteration << "_" <<
			trainingSet <<
			"-trainingSet_" << epoch <<
			"-epoch_" << learningRate <<
			"-learning_" << decayRate << "-decay.csv";
	ofstream _file(fileName.str());

	for (int i = 0; i < map.size(); i++) {
		for (int j = 0; j < connections[i].size(); j++) {
			if (j == (connections[i].size() - 1)) _file << connections[i][j].weight << endl;
			else _file << connections[i][j].weight << ", ";
		}
	}

	_file.close();
}

