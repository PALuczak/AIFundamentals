#ifndef PERCEPTRONWINDOW_H
#define PERCEPTRONWINDOW_H

#include <QDoubleSpinBox>
#include <QMainWindow>
#include <QTableWidget>
#include <QValueAxis>
#include <QtCharts/QChartView>
#include <QtCharts/QLineSeries>
#include <algorithm>
#include <climits>
#include <functional>
#include <random>
#include <string>
#include <tuple>
#include "network.h"
#include "neuron.h"

namespace Ui {
class PerceptronWindow;
}
class PerceptronWindow : public QMainWindow {
  Q_OBJECT

 public:
  explicit PerceptronWindow(QWidget *parent = nullptr);
  ~PerceptronWindow();

  void enableNetwork();

 private slots:
  void on_addRowButton_clicked();
  void on_removeRowButton_clicked();
  void on_calculateButton_clicked();

  void on_betaBox_valueChanged(double arg1);

  void on_functionBox_currentTextChanged(const QString &arg1);

  void on_thetaBox_valueChanged(double arg1);

  void on_trainButton_clicked();

  void on_networkInputAddRowButton_clicked();

  void on_networkInputRemoveRowButton_clicked();

  void on_networkOutputAddRowButton_clicked();

  void on_networkOutputRemoveRowButton_clicked();

  void on_networkTrainButton_clicked();

  void on_networkCalculateButton_clicked();

  void on_networkCreateButton_clicked();

  void on_networkLayerBox_valueChanged(int arg1);

  void on_networkNeuronBox_valueChanged(int arg1);

  void on_networkBetaBox_valueChanged(double arg1);

  void on_networkThetaBox_valueChanged(double arg1);

 private:
  Ui::PerceptronWindow *ui;
  QtCharts::QChart *sigmoidChart;
  const double sigmoidPlotOffset{5};
  const size_t sigmoidPlotPoints{1000};
  std::unique_ptr<NeuralNetwork::NeuralNetwork> network{nullptr};

  QDoubleSpinBox *createNumberCell();
  QDoubleSpinBox *createInputCell();
  QDoubleSpinBox *createWeightCell();
  QMap<QString, std::function<double(double)>> perceptronFunctions;
  std::tuple<std::vector<double>, std::vector<double>> getInputVectors();
  void addTrainingPoint(double value);
  std::vector<double> getNetworkInputVector();
  std::vector<double> getNetworkOutputVector();
  void disableNetwork();
};

#endif  // PERCEPTRONWINDOW_H
