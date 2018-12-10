#include "network.h"

#include <utility>
std::vector<double> NeuralNetwork::NeuralNetwork::simulate(
    std::vector<double> &input) {
  for (auto it = std::begin(layers); it != std::end(layers); ++it) {
    if (it == std::begin(layers)) {
      for (size_t i = 0; i < (*it).neurons.size(); ++i) {
        (*it).neurons.at(i) =
            Neuron::neuron(input, (*it).weights.at(i), neuron);
      }
    } else {
      for (size_t i = 0; i < (*it).neurons.size(); ++i) {
        (*it).neurons.at(i) =
            Neuron::neuron((*(it - 1)).neurons, (*it).weights.at(i), neuron);
      }
    }
  }
  return (*(std::end(layers) - 1)).neurons;
}

std::vector<double> NeuralNetwork::NeuralNetwork::train(
    std::vector<double> &input, std::vector<double> &output, double eta,
    double epsilon, size_t max_iterations) {
  for (size_t j = 0; j < max_iterations; ++j) {
    auto result = this->simulate(input);
    std::vector<double> output_error;
    std::transform(std::begin(output), std::end(output), std::begin(result),
                   std::back_inserter(output_error), std::minus<>());
    auto max_error = *std::max_element(
        std::begin(output_error), std::end(output_error),
        [](auto a, auto b) { return (std::abs(a) < std::abs(b)); });
    if (std::abs(max_error) < epsilon) break;
    for (auto it = std::rbegin(layers); it != std::rend(layers); ++it) {
      std::vector<double> &inputs =
          (it == std::rend(layers) - 1) ? input : (it + 1)->neurons;
      auto &weights = (*it).weights;

      std::vector<double> state_derivative;
      for (size_t i = 0; i < it->neurons.size(); ++i) {
        state_derivative.push_back(
            Neuron::neuron(inputs, weights.at(i), neuronDerivative));
      }

      std::vector<double> projected_error;
      if (it != std::rend(layers) - 1) {
        for (size_t i = 0; i < (it + 1)->neurons.size(); ++i) {
          double val = 0.0;
          for (size_t j = 0; j < weights.size(); ++j) {
            val += output_error.at(j) * state_derivative.at(j) *
                   weights.at(j).at(i);
          }
          projected_error.push_back(val);
        }
      }

      for (size_t i = 0; i < (*it).neurons.size(); ++i) {
        std::vector<double> delta =
            Neuron::detaRule(inputs, weights.at(i), output_error.at(i), neuron,
                             neuronDerivative, eta);
        std::transform(std::begin(delta), std::end(delta),
                       std::begin(weights.at(i)), std::begin(weights.at(i)),
                       std::plus<>());
      }
      output_error = projected_error;
    }
  }
  return this->simulate(input);
}

NeuralNetwork::NeuralNetworkBuilder &
NeuralNetwork::NeuralNetworkBuilder::setTheta(double theta) {
  this->theta = theta;
  return *this;
}

NeuralNetwork::NeuralNetworkBuilder &
NeuralNetwork::NeuralNetworkBuilder::setIntermediateLayers(size_t layers) {
  this->intermediate_layers = layers;
  return *this;
}

NeuralNetwork::NeuralNetworkBuilder &
NeuralNetwork::NeuralNetworkBuilder::setIntermediateNeurons(size_t neurons) {
  this->intermediate_neurons = neurons;
  return *this;
}

NeuralNetwork::NeuralNetworkBuilder &
NeuralNetwork::NeuralNetworkBuilder::setInputNeurons(size_t neurons) {
  this->input_neurons = neurons;
  return *this;
}

NeuralNetwork::NeuralNetworkBuilder &
NeuralNetwork::NeuralNetworkBuilder::setOutputNeurons(size_t neurons) {
  this->output_neurons = neurons;
  return *this;
}

NeuralNetwork::NeuralNetworkBuilder &
NeuralNetwork::NeuralNetworkBuilder::setSigmoid(
    std::function<double(double)> sigmoid) {
  this->sigmoid = std::move(sigmoid);
  return *this;
}

NeuralNetwork::NeuralNetworkBuilder &
NeuralNetwork::NeuralNetworkBuilder::setSigmoidDerivative(
    std::function<double(double)> derivative) {
  this->sigmoidDerivative = std::move(derivative);
  return *this;
}

std::unique_ptr<NeuralNetwork::NeuralNetwork>
NeuralNetwork::NeuralNetworkBuilder::build() {
  std::random_device rand_device;
  std::default_random_engine rand_engine(rand_device());
  std::uniform_real_distribution<double> uniform_dist(-1.0, 1.0);
  auto network = std::make_unique<NeuralNetwork>();
  network->layers.reserve(intermediate_layers + 1);
  network->layers.resize(intermediate_layers + 1);
  for (auto it = std::begin(network->layers); it != std::end(network->layers);
       ++it) {
    if (it == (std::end(network->layers) - 1)) {
      (*it).neurons.resize(output_neurons);
      (*it).weights.resize(output_neurons);
    } else {
      (*it).neurons.resize(intermediate_neurons);
      (*it).weights.resize(intermediate_neurons);
    }
    std::fill(std::begin((*it).neurons), std::end((*it).neurons), 0);
    for (auto &neuron_weights : (*it).weights) {
      if (it == std::begin(network->layers)) {
        neuron_weights.resize(input_neurons);
      } else {
        neuron_weights.resize(intermediate_neurons);
      }
      std::generate(std::begin(neuron_weights), std::end(neuron_weights),
                    [&uniform_dist, &rand_engine]() {
                      return uniform_dist(rand_engine);
                    });
    }
  }
  network->neuron = sigmoid;
  network->neuronDerivative = sigmoidDerivative;
  network->theta = theta;
  return network;
}
