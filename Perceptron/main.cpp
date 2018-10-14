#include <QApplication>
#include "perceptronwindow.h"

int main(int argc, char *argv[]) {
  QApplication a(argc, argv);
  PerceptronWindow w;
  w.show();
  return a.exec();
}
