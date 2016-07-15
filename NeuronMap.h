/*
 * NeuronMap.h
 *
 *  Created on: Jul 14, 2016
 *      Author: trabucco
 */

#ifndef NEURONMAP_H_
#define NEURONMAP_H_

#include "Neuron.h"
#include "Synapse.h"
#include <vector>
#include <math.h>
#include <assert.h>
#include <sstream>
#include <fstream>
#include <iostream>
#include <sys/time.h>
using namespace std;

class NeuronMap {
private:
	double distance(vector<Synapse> s, vector<double> t);
	double neighborhood(int a, int b);
	double width;
	double learningRate;
	double decayRate;
	void activateNeuron(vector<double> input, int dimension, int index);
	void findLargestDistance(vector<double> input, double &bestDistance, int &bestIndex, int dimension, int index);
	void getCorrection(vector<double> input, int bestIndex, int dimension, int index, bool print);
	void updateWeight(int dimension, int index);
public:
	unsigned int type;
	vector<Neuron> map;
	vector<vector<Synapse> > connections;
	int *dim;
	NeuronMap(int connectionsPerNeuron, int *_dim, unsigned int _type, double _width, double _learningRate, double _decayRate);
	virtual ~NeuronMap();
	void online(vector<double> input, bool print);
	void batch(vector<double> input, bool print, bool update);
	vector<double> recognize(vector<double> input);
	void toFile(int iteration, int trainingSet, int epoch);
};

#endif /* NEURONMAP_H_ */
