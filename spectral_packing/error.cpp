#include "error.h"

void raise_and_kill(string msg) {
  cout << msg << endl;
  exit(1);
}

void raise(string msg) {
  throw invalid_argument("An error occurred");
}
