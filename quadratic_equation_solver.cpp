#include <istream>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>

#include "Eigen/Dense"
#include "Eigen/LU"
#include "Eigen/QR"

using namespace std;
using namespace Eigen;

typedef struct coordinate_type {
    double x;
    double y;
} Coordinate;

int main(int argc, char **argv) {

    string inputFile;
    vector<Coordinate> coordinates;
    bool withQR = false;

    // parse arguments
    if (argc <= 1) {
        cout << "No input file parameter was passed." << endl;
        cout << "Default input file: input.txt" << endl << endl;
        inputFile = "input.txt"; // default
    } else {
        if (argc >= 2) {
            string firstParameter = argv[1];
            if (firstParameter == "--qr") {
                cout << "Input file parameter is wrong." << endl;
                cout << "Default input file: input.txt" << endl << endl;
                inputFile = "input.txt"; // default
                withQR = true;
            } else
                inputFile = firstParameter;
        }
        if (argc >= 3) {
            string secondParameter = argv[2];
            if (secondParameter == "--qr")
                withQR = true;
            else {
                cout << "QR parameter is wrong." << endl;
                return 0;
            }
        }
        if (argc >= 4) {
            cout << "Too many parameters were passed." << endl;
            return 0;
        }
    }

    // read input file
    ifstream input(inputFile);
    string input_x, input_y;
    if (input.is_open()) {
        while (input >> input_x >> input_y) {
            Coordinate temp = {
                    stod(input_x),
                    stod(input_y)
            };
            coordinates.push_back(temp);
        }
        input.close();
    } else {
        cout << "Input file can not be found." << endl;
        return 0;
    }

    // initialize equation variables
    int N = coordinates.size();
    int M = 3;
    MatrixXd A(N, M);
    VectorXd y(N);
    VectorXd w(M);

    // construct equation variables with given input
    for (int counter = 0; counter < N; counter++) {
        Coordinate temp = coordinates[counter];
        A(counter, 0) = temp.x * temp.x;
        A(counter, 1) = temp.x;
        A(counter, 2) = 1;
        y(counter) = temp.y;
    }

    cout << "Matrix A:" << endl;
    cout << A << endl << endl;

    cout << "Vector y:" << endl;
    cout << y << endl << endl;

    // check number of given coordinates
    if (N < 3) {
        cout << "There are not enough coordinates." << endl;
        return 0;
    }

    // check square-matrix
    // solve with LU decomposition
    else if (N == 3) {
        MatrixXd P = A.partialPivLu().permutationP();
        cout << "Matrix P:" << endl;
        cout << P << endl << endl;

        MatrixXd L = MatrixXd::Identity(N, M);
        cout << "Matrix L:" << endl;
        cout << L << endl << endl;

        L.triangularView<StrictlyLower>() = A.partialPivLu().matrixLU();
        MatrixXd U = A.partialPivLu().matrixLU().triangularView<Upper>();

        cout << "Matrix U:" << endl;
        cout << U << endl << endl;

        cout << "A = P * L * U: " << endl;
        cout << P * L * U << endl << endl;

        w = A.partialPivLu().solve(y);
    } else {
        // solve with QR decomposition
        if (withQR == 1) {
            MatrixXd Q = A.householderQr().householderQ();
            cout << "Matrix Q:" << endl;
            cout << Q << endl << endl;

            MatrixXd R = Q.transpose() * A;
            cout << "Matrix R:" << endl;
            cout << R << endl << endl;

            cout << "I = Q * Q^T:" << endl;
            cout << Q * Q.transpose() << endl << endl;

            cout << "A = Q * R: " << endl;
            cout << Q * R << endl << endl;

            w = A.householderQr().solve(y);
        }

        // solve with LU decomposition
        else {
            MatrixXd aPrime = A.transpose() * A;
            cout << "Matrix A Prime:" << endl;
            cout << aPrime << endl << endl;

            VectorXd yPrime = A.transpose() * y;
            cout << "Vector Y Prime:" << endl;
            cout << yPrime << endl << endl;

            MatrixXd P = aPrime.partialPivLu().permutationP();
            cout << "Matrix P:" << endl;
            cout << P << endl << endl;

            MatrixXd L = MatrixXd::Identity(M, M);
            cout << "Matrix L:" << endl;
            cout << L << endl << endl;

            L.triangularView<StrictlyLower>() = aPrime.partialPivLu().matrixLU();
            MatrixXd U = aPrime.partialPivLu().matrixLU().triangularView<Upper>();

            cout << "Matrix U:" << endl;
            cout << U << endl << endl;

            cout << "A Prime = P * L * U: " << endl;
            cout << P * L * U << endl << endl;

            w = aPrime.partialPivLu().solve(yPrime);
        }
    }

    cout << "Vector w:" << endl;
    cout << w << endl << endl;

    // show quadratic equation
    double a = w(0);
    double b = w(1);
    double c = w(2);
    cout << "Equation: " << "y = " << a << "*x^2 + " << b << "*x + " << c << endl;

    double sumOfSquaredErrors = 0;
    // calculate squared errors
    for (Coordinate temp: coordinates) {
        double yPrime = a * (temp.x * temp.x) + b * temp.x + c;
        double squaredError = temp.y - yPrime;
        sumOfSquaredErrors += squaredError;
    }
    cout << "Error: " << sumOfSquaredErrors << endl;

    return 0;
}