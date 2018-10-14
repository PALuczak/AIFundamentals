#include "perceptronwindow.h"
#include "ui_perceptronwindow.h"

PerceptronWindow::PerceptronWindow(QWidget* parent)
    : QMainWindow(parent), ui(new Ui::PerceptronWindow) {
  ui->setupUi(this);
  this->perceptronFunctions.insert("heaviside", Perceptron::heaviside);
  this->perceptronFunctions.insert("sign", Perceptron::sign);
  this->perceptronFunctions.insert("logistic", Perceptron::logistic);
  this->perceptronFunctions.insert("tanh", Perceptron::hypertan);
  ui->functionBox->addItems(this->perceptronFunctions.keys());
  ui->thetaBox->setValue(1.0);
  ui->thetaBox->setRange(-qInf(), qInf());
}

PerceptronWindow::~PerceptronWindow() { delete ui; }

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
