#include "perceptronwindow.h"
#include "ui_perceptronwindow.h"

#include <QValueAxis>

PerceptronWindow::PerceptronWindow(QWidget* parent)
    : QMainWindow(parent), ui(new Ui::PerceptronWindow) {
  ui->setupUi(this);
  this->chart = new QtCharts::QChart();
  this->chart->legend()->hide();
  this->chart->createDefaultAxes();
  this->view = new QtCharts::QChartView();
  this->view->setChart(this->chart);
  this->view->setRenderHint(QPainter::Antialiasing);
  ui->plotLayout->addWidget(this->view);
  this->perceptronFunctions.insert("heaviside", Perceptron::heaviside);
  this->perceptronFunctions.insert("sign", Perceptron::sign);
  this->perceptronFunctions.insert("logistic", Perceptron::logistic);
  this->perceptronFunctions.insert("tanh", Perceptron::hypertan);
  ui->functionBox->addItems(this->perceptronFunctions.keys());
  ui->thetaBox->setValue(1.0);
  ui->betaBox->setSingleStep(0.1);
  ui->thetaBox->setRange(-qInf(), qInf());
  ui->betaBox->setValue(1.0);
  ui->betaBox->setSingleStep(0.1);
  ui->betaBox->setRange(-qInf(), qInf());
}

PerceptronWindow::~PerceptronWindow() {
  while (ui->inputTable->rowCount()) {
    int rows = ui->inputTable->rowCount() - 1;
    delete ui->inputTable->takeItem(rows, 0);
    delete ui->inputTable->takeItem(rows, 1);
    ui->inputTable->removeRow(rows);
  }
  delete this->chart;
  delete this->view;
  delete ui;
}

QDoubleSpinBox* PerceptronWindow::createNumberCell() {
  auto cell = new QDoubleSpinBox();
  cell->setDecimals(4);
  cell->setSingleStep(0.1);
  cell->setButtonSymbols(QAbstractSpinBox::NoButtons);
  return cell;
}

QDoubleSpinBox* PerceptronWindow::createInputCell() {
  auto cell = createNumberCell();
  cell->setRange(-1.0, 1.0);
  return cell;
}

QDoubleSpinBox* PerceptronWindow::createWeightCell() {
  auto cell = createNumberCell();
  cell->setRange(-qInf(), qInf());
  return cell;
}

void PerceptronWindow::on_addRowButton_clicked() {
  int rows = ui->inputTable->rowCount();
  ui->inputTable->insertRow(rows);
  ui->inputTable->setCellWidget(rows, 0, createInputCell());
  ui->inputTable->setCellWidget(rows, 1, createWeightCell());
}

void PerceptronWindow::on_removeRowButton_clicked() {
  int rows = ui->inputTable->rowCount() - 1;
  delete ui->inputTable->takeItem(rows, 0);
  delete ui->inputTable->takeItem(rows, 1);
  ui->inputTable->removeRow(rows);
}

void PerceptronWindow::on_calculateButton_clicked() {
  int rows = ui->inputTable->rowCount();
  std::vector<double> inputs;
  std::vector<double> weights;
  inputs.reserve(static_cast<size_t>(rows + 1));
  weights.reserve(static_cast<size_t>(rows + 1));
  inputs.push_back(-1);
  weights.push_back(ui->thetaBox->value());
  // QTableWidget does not allow for proper range based access
  for (auto i = 0; i < rows; ++i) {
    inputs.push_back(
        dynamic_cast<QDoubleSpinBox*>(ui->inputTable->cellWidget(i, 0))
            ->value());
    weights.push_back(
        dynamic_cast<QDoubleSpinBox*>(ui->inputTable->cellWidget(i, 1))
            ->value());
  }
  QString function = ui->functionBox->currentText();
  double activation = Perceptron::perceptron(
      inputs, weights, this->perceptronFunctions.value(function));
  ui->outputText->append(QString("Activation with %1 bound: %2")
                             .arg(function, QString::number(activation)));
}

void PerceptronWindow::on_betaBox_valueChanged(double arg1) {
  Perceptron::beta = arg1;
  on_functionBox_currentTextChanged(nullptr);
}

void PerceptronWindow::on_functionBox_currentTextChanged(const QString&) {
  this->chart->removeAllSeries();
  QtCharts::QLineSeries* series = new QtCharts::QLineSeries();
  double theta = ui->thetaBox->value();
  std::vector<double> xvals(plotPoints);
  std::generate(std::begin(xvals), std::end(xvals),
                [n = theta - plotOffset]() mutable { return n += 0.01; });
  QString function = ui->functionBox->currentText();
  for (auto i : xvals) {
    series->append(i, this->perceptronFunctions.value(function)(i - theta));
  }
  this->chart->addSeries(series);
  this->chart->createDefaultAxes();
  this->view->update();
}

void PerceptronWindow::on_thetaBox_valueChanged(double) {
  on_functionBox_currentTextChanged(nullptr);
}
