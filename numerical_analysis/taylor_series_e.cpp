#include <iostream>
#include <string>
#include <math.h>

double factorial(int n) {
    // Factorial cannot be negative
    if (n < 0) {
        std::cerr << "Error: <n> must be a non-negative integer" << std::endl;
        return 1;
    }

    double result = 1.0;

    for (int i = 2; i <= n; i++) {
        result *= i;
    }

    return result;
}

double e_approximation(double x, int n) {
    double approximation = 0.0;

    for (int i = 0; i < n; i++) {
        double term = (pow(x, i) / factorial(i));
        approximation += term;
    }

    return approximation;
}

int main(int argc, char** argv) {
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0] << " <x> <n> <precision>" << std::endl;
        return 1;
    }

    try {
        double x = std::stod(argv[1]);
        int n = std::stoi(argv[2]);
        int precision = std::stoi(argv[3]);

        if (precision < 1) {
            std::cerr << "Error: <precision> must be greater than 0" << std::endl;
            return 1;
        }

        double result = e_approximation(x, n);

        std::cout.precision(precision);
        std::cout << "Approximation for e^" << x << " for " << n << " terms: " << result << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Error: Invalid command line argument. Provide numeric values for <x> (double), <n> (int), and <precision> (int)." << std::endl;
        return 1;
    }

    return 0;
}

