//============================================================================
// Name        : main.cpp
// Author      : Brandon Trabucco
// Version     : 1.0.5
// Copyright   : This project is licensed under the GNU General Public License
// Description : This project is a test implementation of a Neural Network
//============================================================================

#include "ImageLoader.h"
#include "NeuronMap.h"
#include <fstream>
#include <vector>
#include <sys/time.h>
#include <stdlib.h>
#include <string.h>
#include <sstream>
#include <numeric>

#define CONVERGENCE_TEST false
using namespace std;

long long getMSec() {
	struct timeval tp;
	gettimeofday(&tp, NULL);
	return tp.tv_sec * 1000 + (tp.tv_usec / 1000);
}

struct tm *getDate() {
	time_t t = time(NULL);
	struct tm *timeObject = localtime(&t);
	return timeObject;
}

int main(int argc, char *argv[]) {
	cout << "Program initializing" << endl;
	if (!(argc > 5)) {
		cout << argv[0] << " <o | b> <training> <epoch> <learning> <decay> <dimensions ...>" << endl;
		return -1;
	}


	/**
	 *
	 * 	Declare the global variables
	 * 	These govern functionality of the program
	 *
	 */
	vector<vector<double> > trainingImages, testImages;
	vector<double> trainingLabels, testLabels;
	int numberImages = 0;
	int imageSize = 0;
	int numberLabels = 0;
	long long networkStart, networkEnd, iterationStart;
	int trainingSize = atoi(argv[2]);
	int epoch = (atoi(argv[3]) < 100) ? 100 : atoi(argv[3]);
	int updatePoints = 100;
	int savePoints = 10;
	double learningRate = atof(argv[4]), decay = atof(argv[5]);
	bool enableBatch = (argv[1][0] == 'b');


	/**
	 *
	 * 	Load the MNIST Dataset
	 * 	To feed into Self Organizing Map
	 *
	 */
	// load training images and labels
	networkStart = getMSec();
	trainingImages = ImageLoader::readMnistImages("/u/trabucco/Desktop/MNIST_Bytes/train-images.idx3-ubyte", numberImages, imageSize);
	trainingLabels = ImageLoader::readMnistLabels("/u/trabucco/Desktop/MNIST_Bytes/train-labels.idx1-ubyte", numberLabels);
	// load test images and labels
	testImages = ImageLoader::readMnistImages("/u/trabucco/Desktop/MNIST_Bytes/t10k-images.idx3-ubyte", numberImages, imageSize);
	testLabels = ImageLoader::readMnistLabels("/u/trabucco/Desktop/MNIST_Bytes/t10k-labels.idx1-ubyte", numberLabels);
	networkEnd = getMSec();
	cout << "Training images and labels loaded in " << (networkEnd - networkStart) << " msecs" << endl;


	/**
	 *
     * 	Initialize the Self Organizing Map
 	 *
   	 */
	int *size = (int *)malloc(sizeof(int) * (argc - 6));
	for (int i = 0; i < (argc - 6); i++) size[i] = atoi(argv[6 + i]);
	NeuronMap map = NeuronMap(imageSize, size, (argc - 6), 10.0, learningRate, decay);


	/**
	 *
	 * 	Iterate through the training and test datasets
	 * 	Output performance information to data files
	 *
	 */
	if (!enableBatch) {
		/**
		*
		* 	This section is for online gradient descent
		*
		*/
		for (int r = 0; r < epoch; r++) {
			if (!(r % (epoch / updatePoints))) iterationStart = getMSec();
			for (int i = 0; i < trainingSize; i++) {
				networkStart = getMSec();
				map.online(trainingImages[CONVERGENCE_TEST ? 0 : i], (i == (trainingSize - 1)) && !(r % (epoch / updatePoints)));
				networkEnd = getMSec();
			} if (!((r+1) % (epoch / updatePoints))) {
				cout << "Epoch " << (r+1) << " " << (((getMSec() - iterationStart))) <<
						"msecs, ETA " << (((((getMSec() - iterationStart)))) * ((double)(epoch) - (double)r) / 1000.0 / 60.0) << "min" << endl;
			} if (!((r+1) % (epoch / savePoints))) {
				map.toFile((r+1), trainingSize, epoch);
			}
		}
	} else {
		/**
		 *
		 * 	This section is for batch gradient descent
		 *
		 */
		for (int r = 0; r < epoch; r++) {
			iterationStart = getMSec();
			for (int i = 0; i < trainingSize; i++) {
				networkStart = getMSec();
				map.batch(trainingImages[CONVERGENCE_TEST ? 0 : i], (i == (trainingSize - 1)) && !(r % (epoch / updatePoints)), (i == (trainingSize - 1)));
				networkEnd = getMSec();
			} if (!((r+1) % (epoch / updatePoints))) {
				cout << "Epoch " << (r+1) << " " << (((getMSec() - iterationStart))) <<
						"msecs, ETA " << (((((getMSec() - iterationStart)))) * ((double)(epoch) - (double)r) / 1000.0 / 60.0) << "min" << endl;
			} if (!((r+1) % (epoch / savePoints))) {
				map.toFile((r+1), trainingSize, epoch);
			}
		}
	}

	vector<double> temp = map.recognize(trainingImages[0]);
	for (int i = 0; i < temp.size(); i++) {
		cout << "Output Neuron " << i << " activating by " << temp[i] << endl;
	}

	cout << "Program finishing" << endl;

	return 0;
}
