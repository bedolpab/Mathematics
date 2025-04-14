#include <iostream>
#include <math.h>

using namespace std;

double factorial(int n) {
  double result = 1.0;

  for(int i=2; i<=n; i++){
    result *= i;
  }

  return result;
}


double sine_approximation(double x, int n) {
  double sum  = 0.0;

  for(int i = 0; i < n; i++) {
    double term = pow(-1, i) * pow(x, (2 * i) + 1) / factorial((2 * i) + 1);
    sum += term;
  }

  return sum;


}

int main() {
  cout << sine_approximation(1.0, 10) << endl;
  return 0;
}
