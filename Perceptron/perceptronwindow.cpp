#include "perceptronwindow.h"
#include "ui_perceptronwindow.h"

#include <QValueAxis>

PerceptronWindow::PerceptronWindow(QWidget* parent)
    : QMainWindow(parent), ui(new Ui::PerceptronWindow) {
  ui->setupUi(this);
  this->sigmoidChart = new QtCharts::QChart();
  this->sigmoidChart->legend()->hide();
  this->sigmoidChart->createDefaultAxes();
  ui->sigmoidView->setChart(this->sigmoidChart);
  ui->sigmoidView->setRenderHint(QPainter::Antialiasing);
  ui->sigmoidView->setBackgroundBrush(Qt::white);

  this->perceptronFunctions.insert("heaviside", Neuron::heaviside);
  this->perceptronFunctions.insert("sign", Neuron::sign);
  this->perceptronFunctions.insert("logistic", Neuron::logistic);
  this->perceptronFunctions.insert("tanh", Neuron::hypertan);

  ui->functionBox->addItems(this->perceptronFunctions.keys());
  ui->functionBox->setCurrentIndex(1);
  ui->thetaBox->setValue(1.0);
  ui->thetaBox->setSingleStep(0.1);
  ui->thetaBox->setRange(-qInf(), qInf());
  ui->betaBox->setValue(1.0);
  ui->betaBox->setSingleStep(0.1);
  ui->betaBox->setRange(-qInf(), qInf());
  ui->expectedBox->setValue(1.0);
  ui->expectedBox->setSingleStep(0.1);
  ui->expectedBox->setRange(-qInf(), qInf());
  ui->learningRateBox->setValue(1);
  ui->learningRateBox->setSingleStep(0.1);
  ui->learningRateBox->setRange(-qInf(), qInf());
  ui->maxIterBox->setValue(99);
  ui->maxIterBox->setSingleStep(1);
  ui->maxIterBox->setRange(1, INT_MAX);
  ui->epsilonBox->setValue(0.1);
  ui->epsilonBox->setSingleStep(0.01);
  ui->epsilonBox->setRange(0, qInf());
}

PerceptronWindow::~PerceptronWindow() {
  while (ui->inputTable->rowCount()) {
    int rows = ui->inputTable->rowCount() - 1;
    delete ui->inputTable->takeItem(rows, 0);
    delete ui->inputTable->takeItem(rows, 1);
    ui->inputTable->removeRow(rows);
  }
  delete this->sigmoidChart;
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

std::tuple<std::vector<double>, std::vector<double>>
PerceptronWindow::getInputVectors() {
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
  return std::make_tuple(inputs, weights);
}

void PerceptronWindow::on_calculateButton_clicked() {
  std::vector<double> inputs;
  std::vector<double> weights;
  std::tie(inputs, weights) = getInputVectors();
  QString function = ui->functionBox->currentText();
  double activation = Neuron::neuron(inputs, weights,
                                     this->perceptronFunctions.value(function));
  ui->outputText->append(QString("Activation with %1 bound: %2")
                             .arg(function, QString::number(activation)));
}

void PerceptronWindow::on_betaBox_valueChanged(double arg1) {
  Neuron::beta = arg1;
  on_functionBox_currentTextChanged(nullptr);
}

void PerceptronWindow::on_functionBox_currentTextChanged(const QString&) {
  if (ui->functionBox->currentText() != "logistic") {
    ui->betaBox->setDisabled(true);
    ui->trainButton->setDisabled(true);
  } else {
    ui->betaBox->setEnabled(true);
    ui->trainButton->setEnabled(true);
  }
  this->sigmoidChart->removeAllSeries();
  QtCharts::QLineSeries* series = new QtCharts::QLineSeries();
  double theta = ui->thetaBox->value();
  std::vector<double> xvals(sigmoidPlotPoints);
  std::generate(
      std::begin(xvals), std::end(xvals),
      [n = theta - sigmoidPlotOffset]() mutable { return n += 0.01; });
  QString function = ui->functionBox->currentText();
  for (auto i : xvals) {
    series->append(i, this->perceptronFunctions.value(function)(i - theta));
  }
  this->sigmoidChart->addSeries(series);
  this->sigmoidChart->createDefaultAxes();
  ui->sigmoidView->update();
}

void PerceptronWindow::on_thetaBox_valueChanged(double) {
  on_functionBox_currentTextChanged(nullptr);
}

void PerceptronWindow::addTrainingPoint(double value) {
  ui->outputText->append(
      QString("Error after iteration: %1").arg(QString::number(value)));
}

void PerceptronWindow::on_trainButton_clicked() {
  std::vector<double> inputs;
  std::vector<double> weights;
  std::tie(inputs, weights) = getInputVectors();

  double expected = ui->expectedBox->value();
  int maxIterations = ui->maxIterBox->value();
  double epsilon = ui->epsilonBox->value();
  double eta = ui->learningRateBox->value();

  auto trainedWeights = Neuron::trainNeuron(
      weights, expected,
      [inputs, weights, expected, eta](const auto& w) {
        return Neuron::detaRule(inputs, w, expected, Neuron::logistic,
                                Neuron::logisticDerivative, eta);
      },
      [inputs](const auto& w) {
        return Neuron::neuron(inputs, w, Neuron::logistic);
      },
      epsilon, maxIterations,
      [this](double err) { this->addTrainingPoint(err); });

  std::string textWeights;
  for (auto val : trainedWeights) {
    textWeights += std::to_string(val);
    textWeights += ", ";
  }

  ui->outputText->append(
      QString("Trained the neuron and set proper weight values %1")
          .arg(QString::fromStdString(textWeights)));
  ui->thetaBox->setValue(trainedWeights.at(0));

  auto weightIt = std::begin(trainedWeights) + 1;
  int rows = ui->inputTable->rowCount();
  for (auto i = 0; i < rows; ++i) {
    dynamic_cast<QDoubleSpinBox*>(ui->inputTable->cellWidget(i, 1))
        ->setValue(*weightIt++);
  }
}
