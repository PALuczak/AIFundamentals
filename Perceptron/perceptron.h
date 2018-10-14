#ifndef PERCEPTRON_H
#define PERCEPTRON_H
#include <cmath>
#include <numeric>
#include <vector>
namespace Perceptron {
inline double heaviside(double in) { return in >= 0.0; }
inline double sign(double in) {
  if (in > 0) return 1.0;
  if (in < 0) return -1.0;
  return 0.0;
}
static double beta = 1.0;
inline double logistic(double in) {
  return 1.0 / (1 + std::exp(-1.0 * in * beta));
}
inline double hypertan(double in) { return std::tanh(in); }

template <typename Lambda>
double perceptron(std::vector<double>& inputs, std::vector<double>& weights,
                  Lambda bound) {
  return bound(std::inner_product(std::begin(inputs), std::end(inputs),
                                  std::begin(weights), 0.0f));
}
}  // namespace Perceptron
#endif  // PERCEPTRON_H
