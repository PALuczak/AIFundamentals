#ifndef NETWORK_H
#define NETWORK_H
#include <algorithm>
#include <functional>
#include <memory>
#include <random>
#include <vector>
#include "neuron.h"
namespace NeuralNetwork {
class NeuralNetworkBuilder;
class NeuralNetwork {
 public:
  std::vector<double> simulate(std::vector<double> &input);
  void train(std::vector<std::vector<double> > &input,
                            std::vector<std::vector<double> > &output, double eta,
                            double epsilon, size_t max_iterations);

  double test(std::vector<double> &input, std::vector<double> &output);
private:
  friend class NeuralNetworkBuilder;
  struct NetworkLayer {
    std::vector<double> neurons;
    std::vector<std::vector<double>> weights;
  };
  std::vector<NetworkLayer> layers;
  double theta{0.0};

  std::function<double(double)> neuron;
  std::function<double(double)> neuronDerivative;
};

class NeuralNetworkBuilder {
 public:
  NeuralNetworkBuilder &setTheta(double theta);
  NeuralNetworkBuilder &setIntermediateLayers(size_t layers);
  NeuralNetworkBuilder &setIntermediateNeurons(size_t neurons);
  NeuralNetworkBuilder &setInputNeurons(size_t neurons);
  NeuralNetworkBuilder &setOutputNeurons(size_t neurons);
  NeuralNetworkBuilder &setSigmoid(std::function<double(double)> sigmoid);
  NeuralNetworkBuilder &setSigmoidDerivative(
      std::function<double(double)> derivative);
  std::unique_ptr<NeuralNetwork> build();

 private:
  size_t intermediate_layers;
  size_t intermediate_neurons;
  size_t input_neurons;
  size_t output_neurons;
  double theta {0.0};
  std::function<double(double)> sigmoid;
  std::function<double(double)> sigmoidDerivative;
};
}  // namespace NeuralNetwork
#endif  // NETWORK_H
