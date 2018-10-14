#ifndef PERCEPTRONWINDOW_H
#define PERCEPTRONWINDOW_H

#include <QDoubleSpinBox>
#include <QMainWindow>
#include <QTableWidget>
#include <QtCharts/QChartView>
#include <QtCharts/QLineSeries>
#include <algorithm>
#include <functional>
#include "perceptron.h"

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

 private:
  Ui::PerceptronWindow *ui;
  QDoubleSpinBox *createNumberCell();
  QDoubleSpinBox *createInputCell();
  QDoubleSpinBox *createWeightCell();
  QMap<QString, std::function<double(double)>> perceptronFunctions;
  QtCharts::QChart *chart;
  QtCharts::QChartView *view;
  const double plotOffset = 5;
  const size_t plotPoints = 1000;
};

#endif  // PERCEPTRONWINDOW_H
