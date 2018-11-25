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
#include "neuron.h"

namespace Ui {
class PerceptronWindow;
}
class PerceptronWindow : public QMainWindow {
  Q_OBJECT

 public:
  explicit PerceptronWindow(QWidget *parent = nullptr);
  ~PerceptronWindow();

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

 private:
  Ui::PerceptronWindow *ui;
  QDoubleSpinBox *createNumberCell();
  QDoubleSpinBox *createInputCell();
  QDoubleSpinBox *createWeightCell();
  QMap<QString, std::function<double(double)>> perceptronFunctions;
  QtCharts::QChart *sigmoidChart;
  const double sigmoidPlotOffset = 5;
  const size_t sigmoidPlotPoints = 1000;
  std::tuple<std::vector<double>, std::vector<double>> getInputVectors();
  void addTrainingPoint(double value);
  std::vector<double> getNetworkInputVector();
  std::vector<double> getNetworkOutputVector();
};

#endif  // PERCEPTRONWINDOW_H
