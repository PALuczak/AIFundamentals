#ifndef NEURON_H
#define NEURON_H
#include <algorithm>
#include <cmath>
#include <numeric>
#include <vector>
namespace Neuron {
inline double heaviside(double in) { return in >= 0.0; }
inline double sign(double in) {
  if (in > 0) return 1.0;
  if (in < 0) return -1.0;
  return 0.0;
}
static volatile double beta = 1.0;
inline double logistic(double in) {
  return 1.0 / (1 + std::exp(-1.0 * in * beta));
}
inline double hypertan(double in) { return std::tanh(in); }

template <typename Lambda>
double neuron(const std::vector<double>& inputs,
              const std::vector<double>& weights, Lambda sigmoid) {
  return sigmoid(std::inner_product(std::begin(inputs), std::end(inputs),
                                    std::begin(weights), 0.0f));
}

inline double logisticDerivative(double in) {
  return beta * logistic(in) * logistic(1.0 - in);
}

template <typename Lambda>
inline std::vector<double> detaRule(const std::vector<double>& inputs,
                                    const std::vector<double>& weights,
                                    double expected, Lambda sigmoid,
                                    Lambda derivative, double rate = 1.0) {
  std::vector<double> deltaWeight;
  deltaWeight.reserve(weights.size());
  std::transform(
      std::begin(inputs), std::end(inputs), std::back_inserter(deltaWeight),
      [rate, expected, weights, inputs, sigmoid, derivative](double input) {
        return rate * (expected - neuron(inputs, weights, sigmoid)) *
               neuron(inputs, weights, derivative) * input;
      });
  return deltaWeight;
}

template <typename Lambda, typename Neuron, typename Callback>
inline std::vector<double> trainNeuron(const std::vector<double>& weights,
                                       double expected, Lambda ruleFunction,
                                       Neuron neuron, double limit = 0.5,
                                       int maxIterations = 100,
                                       Callback onIterate = []() {}) {
  std::vector<double> finalWeights = weights;
  volatile double error = 1.0;
  for (int i = 0; i < maxIterations; ++i) {
    auto deltaWeight = ruleFunction(finalWeights);
    std::transform(std::begin(finalWeights), std::end(finalWeights),
                   std::begin(deltaWeight), std::begin(finalWeights),
                   [](double old, double delta) { return old + delta; });
    error = std::abs(expected - neuron(finalWeights));
    onIterate(error);
    if (error < limit) return finalWeights;
  }
  return finalWeights;
}
}  // namespace Neuron
#endif  // NEURON_H
