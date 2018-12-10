#include "perceptronwindow.h"
#include "ui_perceptronwindow.h"

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
  ui->expectedBox->setRange(-1, 1);
  ui->learningRateBox->setValue(1);
  ui->learningRateBox->setSingleStep(0.1);
  ui->learningRateBox->setRange(-qInf(), qInf());
  ui->maxIterBox->setValue(99);
  ui->maxIterBox->setSingleStep(1);
  ui->maxIterBox->setRange(1, INT_MAX);
  ui->epsilonBox->setValue(0.1);
  ui->epsilonBox->setSingleStep(0.01);
  ui->epsilonBox->setRange(0, qInf());

  ui->networkLayerBox->setValue(0);
  ui->networkLayerBox->setSingleStep(1);
  ui->networkLayerBox->setRange(0, INT_MAX);
  ui->networkNeuronBox->setValue(1);
  ui->networkNeuronBox->setSingleStep(1);
  ui->networkNeuronBox->setRange(1, INT_MAX);
  ui->networkBetaBox->setValue(1.0);
  ui->networkBetaBox->setSingleStep(0.1);
  ui->networkBetaBox->setRange(-qInf(), qInf());
  ui->networkThetaBox->setValue(0.0);
  ui->networkThetaBox->setSingleStep(0.1);
  ui->networkThetaBox->setRange(-qInf(), qInf());
  ui->networkEtaBox->setValue(1.0);
  ui->networkEtaBox->setSingleStep(0.1);
  ui->networkEtaBox->setRange(-qInf(), qInf());
  ui->networkEpsilonBox->setValue(0.1);
  ui->networkEpsilonBox->setSingleStep(0.01);
  ui->networkEpsilonBox->setRange(-qInf(), qInf());
  ui->networkIterationsBox->setValue(99);
  ui->networkIterationsBox->setSingleStep(1);
  ui->networkIterationsBox->setRange(1, INT_MAX);

  disableNetwork();
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
  std::random_device r;
  std::default_random_engine re(r());
  std::uniform_real_distribution<double> uniform_dist(-1.0, 1.0);
  auto cell = createNumberCell();
  cell->setRange(-1.0, 1.0);
  cell->setValue(uniform_dist(re));
  return cell;
}

QDoubleSpinBox* PerceptronWindow::createWeightCell() {
  std::random_device r;
  std::default_random_engine re(r());
  std::uniform_real_distribution<double> uniform_dist(-2.0, 2.0);
  auto cell = createNumberCell();
  cell->setRange(-qInf(), qInf());
  cell->setValue(uniform_dist(re));
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
  auto* series = new QtCharts::QLineSeries();
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

void PerceptronWindow::on_networkInputAddRowButton_clicked() {
  int rows = ui->networkInputTable->rowCount();
  ui->networkInputTable->insertRow(rows);
  ui->networkInputTable->setCellWidget(rows, 0, createInputCell());
  disableNetwork();
}

void PerceptronWindow::on_networkInputRemoveRowButton_clicked() {
  int rows = ui->networkInputTable->rowCount() - 1;
  delete ui->networkInputTable->takeItem(rows, 0);
  ui->networkInputTable->removeRow(rows);
  disableNetwork();
}

std::vector<double> PerceptronWindow::getNetworkInputVector() {
  int rows = ui->networkInputTable->rowCount();
  std::vector<double> inputs;
  inputs.reserve(static_cast<size_t>(rows + 1));
  // QTableWidget does not allow for proper range based access
  for (auto i = 0; i < rows; ++i) {
    inputs.push_back(
        dynamic_cast<QDoubleSpinBox*>(ui->networkInputTable->cellWidget(i, 0))
            ->value());
  }
  return inputs;
}

void PerceptronWindow::on_networkOutputAddRowButton_clicked() {
  int rows = ui->networkOutputTable->rowCount();
  ui->networkOutputTable->insertRow(rows);
  ui->networkOutputTable->setCellWidget(rows, 0, createInputCell());
  disableNetwork();
}

void PerceptronWindow::disableNetwork() {
  ui->networkCalculateButton->setDisabled(true);
  ui->networkTrainButton->setDisabled(true);
}

void PerceptronWindow::on_networkOutputRemoveRowButton_clicked() {
  int rows = ui->networkOutputTable->rowCount() - 1;
  delete ui->networkOutputTable->takeItem(rows, 0);
  ui->networkOutputTable->removeRow(rows);
  disableNetwork();
}

std::vector<double> PerceptronWindow::getNetworkOutputVector() {
  int rows = ui->networkOutputTable->rowCount();
  std::vector<double> outputs;
  outputs.reserve(static_cast<size_t>(rows + 1));
  // QTableWidget does not allow for proper range based access
  for (auto i = 0; i < rows; ++i) {
    outputs.push_back(
        dynamic_cast<QDoubleSpinBox*>(ui->networkOutputTable->cellWidget(i, 0))
            ->value());
  }
  return outputs;
}

void PerceptronWindow::on_networkTrainButton_clicked() {
  auto inputs = getNetworkInputVector();
  auto outputs = getNetworkOutputVector();
  auto eta = ui->networkEtaBox->value();
  auto iterations = static_cast<size_t>(ui->networkIterationsBox->value());
  auto epsilon = ui->networkEpsilonBox->value();
  auto result = network->train(inputs, outputs, eta, epsilon, iterations);
  std::string printable_result = "Network trained, resulting output: ";
  for (auto& val : result) {
    printable_result.append(std::to_string(val) + ", ");
  }
  printable_result.append(" final errors: ");
  std::vector<double> err;
  std::transform(std::begin(outputs), std::end(outputs), std::begin(result),
                 std::back_inserter(err), std::minus<>());
  for (auto& val : err) {
    printable_result.append(std::to_string(val) + ", ");
  }
  ui->outputText->append(QString::fromStdString(printable_result));
}

void PerceptronWindow::on_networkCalculateButton_clicked() {
  auto inputs = getNetworkInputVector();
  auto result = network->simulate(inputs);
  std::string printable_result = "Network output: ";
  for (auto& val : result) {
    printable_result.append(std::to_string(val) + ", ");
  }
  ui->outputText->append(QString::fromStdString(printable_result));
}

void PerceptronWindow::enableNetwork() {
  ui->networkCalculateButton->setEnabled(true);
  ui->networkTrainButton->setEnabled(true);
}

void PerceptronWindow::on_networkCreateButton_clicked() {
  NeuralNetwork::NeuralNetworkBuilder builder;
  auto network_inputs = static_cast<size_t>(ui->networkInputTable->rowCount());
  auto network_outputs =
      static_cast<size_t>(ui->networkOutputTable->rowCount());
  auto intermediate_layers = static_cast<size_t>(ui->networkLayerBox->value());
  auto intermediate_neurons =
      static_cast<size_t>(ui->networkNeuronBox->value());
  this->network = builder.setInputNeurons(network_inputs)
                      .setOutputNeurons(network_outputs)
                      .setIntermediateLayers(intermediate_layers)
                      .setIntermediateNeurons(intermediate_neurons)
                      .setSigmoid(Neuron::logistic)
                      .setSigmoidDerivative(Neuron::logisticDerivative)
                      .build();
  Neuron::beta = ui->networkBetaBox->value();
  enableNetwork();
}

void PerceptronWindow::on_networkLayerBox_valueChanged(int arg1) {
  (void)arg1;
  disableNetwork();
}

void PerceptronWindow::on_networkNeuronBox_valueChanged(int arg1) {
  (void)arg1;
  disableNetwork();
}

void PerceptronWindow::on_networkBetaBox_valueChanged(double arg1) {
  (void)arg1;
  disableNetwork();
}

void PerceptronWindow::on_networkThetaBox_valueChanged(double arg1) {
  (void)arg1;
  disableNetwork();
}
