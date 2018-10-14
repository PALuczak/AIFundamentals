#ifndef PERCEPTRONWINDOW_H
#define PERCEPTRONWINDOW_H

#include <QDoubleSpinBox>
#include <QMainWindow>
#include <QTableWidget>
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

 private:
  Ui::PerceptronWindow *ui;
  QDoubleSpinBox *createNumberCell();
  QDoubleSpinBox *createInputCell();
  QDoubleSpinBox *createWeightCell();
  QMap<QString, std::function<double(double)>> perceptronFunctions;
};

#endif  // PERCEPTRONWINDOW_H
