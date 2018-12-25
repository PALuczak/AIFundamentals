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
  ui->thetaBox->setSingleStep(0.01);
  ui->thetaBox->setRange(-qInf(), qInf());
  ui->betaBox->setValue(1.0);
  ui->betaBox->setSingleStep(0.01);
  ui->betaBox->setRange(-qInf(), qInf());
  ui->expectedBox->setValue(1.0);
  ui->expectedBox->setSingleStep(0.1);
  ui->expectedBox->setRange(-1, 1);
  ui->learningRateBox->setValue(0.5);
  ui->learningRateBox->setSingleStep(0.01);
  ui->learningRateBox->setRange(-qInf(), qInf());
  ui->maxIterBox->setValue(99);
  ui->maxIterBox->setSingleStep(1);
  ui->maxIterBox->setRange(1, INT_MAX);
  ui->epsilonBox->setValue(0.01);
  ui->epsilonBox->setSingleStep(0.001);
  ui->epsilonBox->setRange(0, qInf());

  ui->networkLayerBox->setValue(3);
  ui->networkLayerBox->setSingleStep(1);
  ui->networkLayerBox->setRange(0, INT_MAX);
  ui->networkNeuronBox->setValue(2);
  ui->networkNeuronBox->setSingleStep(1);
  ui->networkNeuronBox->setRange(1, INT_MAX);
  ui->networkBetaBox->setValue(1.0);
  ui->networkBetaBox->setSingleStep(0.1);
  ui->networkBetaBox->setRange(-qInf(), qInf());
  ui->networkThetaBox->setValue(0.0);
  ui->networkThetaBox->setSingleStep(0.1);
  ui->networkThetaBox->setRange(-qInf(), qInf());
  ui->networkEtaBox->setValue(0.01);
  ui->networkEtaBox->setSingleStep(0.001);
  ui->networkEtaBox->setRange(-qInf(), qInf());
  ui->networkEpsilonBox->setValue(0.001);
  ui->networkEpsilonBox->setSingleStep(0.001);
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

// neural network handling below

void PerceptronWindow::on_networkTrainButton_clicked() {
  // go through the train set
  auto inputs = this->inputTrain;
  auto outputs = this->outputTrain;
  auto eta = ui->networkEtaBox->value();
  auto iterations = static_cast<size_t>(ui->networkIterationsBox->value());
  auto epsilon = ui->networkEpsilonBox->value();
  network->train(inputs, outputs, eta, epsilon, iterations);
  ui->outputText->append("Network trained");
}

void PerceptronWindow::on_networkCalculateButton_clicked() {
  // go through the test set
  auto inputs = this->inputTest;
  auto outputs = this->outputTest;
  double accuracy = 0.0;
  for (size_t i = 0; i < inputs.size(); ++i) {
    auto result = network->test(inputs.at(i), outputs.at(i));
    accuracy += result;
//    ui->outputText->append(
//        QString("Network tested, error for a sample: %1 %").arg(result * 100.0));
  }
  ui->outputText->append(
      QString("Network tested, average error: %1 %")
          .arg((accuracy/inputs.size()) * 100.0));
}

void PerceptronWindow::enableNetwork() {
  ui->networkCalculateButton->setEnabled(true);
  ui->networkTrainButton->setEnabled(true);
}

void PerceptronWindow::disableNetwork() {
  ui->networkCalculateButton->setDisabled(true);
  ui->networkTrainButton->setDisabled(true);
}

void PerceptronWindow::on_networkCreateButton_clicked() {
  NeuralNetwork::NeuralNetworkBuilder builder;
  auto network_inputs = static_cast<size_t>(ui->networkInputsBox->value());
  auto network_outputs = static_cast<size_t>(ui->networkClassesBox->value());
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

  // generate set and split
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> dis(0.0, 1000.0 * ui->networkClassesBox->value());
  std::vector<std::vector<double>> centers;
  std::generate_n(std::back_inserter(centers), ui->networkClassesBox->value(), [&](){
      std::vector<double> center;
      std::generate_n(std::back_inserter(center), ui->networkInputsBox->value(), [&](){
          return dis(gen);
      });
      return center;
  });

  this->inputData.clear();
  this->outputData.clear();
  for (int j = 0; j < ui->networkSamplesBox->value(); ++j){
      for (size_t i = 0; i < centers.size(); ++i) {
          std::vector<double> sample;
          std::for_each(std::begin(centers.at(i)), std::end(centers.at(i)), [&](auto val){
              std::normal_distribution<> gauss{val, 3.0};
              sample.push_back(gauss(gen));
          });
          this->inputData.push_back(sample);

          std::vector<double> classification;
          for (size_t j = 0; j < centers.size(); ++j) {
              if(i==j) classification.push_back(1);
              else classification.push_back(0);
          }
          this->outputData.push_back(classification);
      }
  }
  this->shuffleAndSplitData();
  enableNetwork();
  ui->outputText->append("Network and sample data created");
}

void PerceptronWindow::shuffleAndSplitData() {
    std::vector<size_t> indexes(inputData.size());
    std::iota(std::begin(indexes), std::end(indexes), 0);
    std::random_shuffle(std::begin(indexes), std::end(indexes));

    std::vector<std::vector<double>> oldInputData;
    std::vector<std::vector<double>> oldOutputData;
    std::copy(std::begin(inputData), std::end(inputData), std::back_inserter(oldInputData));
    std::copy(std::begin(outputData), std::end(outputData), std::back_inserter(oldOutputData));

    for (size_t i = 0; i < indexes.size(); ++i) {
        inputData.at(i) = oldInputData.at(indexes.at(i));
        outputData.at(i) = oldOutputData.at(indexes.at(i));
    }

    auto split_offset = static_cast<long long>(std::round(
                                                   std::distance(std::begin(inputData), std::end(inputData))
                                                   * ui->networkSplitRatioBox->value()));

    inputTrain.clear();
    inputTest.clear();
    outputTrain.clear();
    outputTest.clear();
    std::copy(std::begin(inputData), std::begin(inputData) + split_offset, std::back_inserter(inputTrain));
    std::copy(std::begin(inputData) + split_offset, std::end(inputData), std::back_inserter(inputTest));
    std::copy(std::begin(outputData), std::begin(outputData) + split_offset, std::back_inserter(outputTrain));
    std::copy(std::begin(outputData) + split_offset, std::end(outputData), std::back_inserter(outputTest));
}
