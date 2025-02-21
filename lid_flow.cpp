#include <iostream>
#include <vector>
#include <fstream>
#include <cmath>
#include <omp.h> // For parallel computing

using namespace std;

// Grid and fluid parameters
const int nx = 41, ny = 41;  // Grid size
const double lx = 2.0, ly = 2.0;
const double dx = lx / (nx - 1);
const double dy = ly / (ny - 1);
const double dt = 0.001;
const int nt = 500;  // Number of time steps
const int nit = 50;  // Iterations for pressure solver
const double rho = 1.0;
const double nu = 0.1;

// Function to solve the Pressure Poisson Equation (PPE)
void pressure_poisson(vector<vector<double>> &p, vector<vector<double>> &b) {
    for (int iter = 0; iter < nit; iter++) {
        vector<vector<double>> pn = p;

        #pragma omp parallel for collapse(2)
        for (int i = 1; i < ny - 1; i++) {
            for (int j = 1; j < nx - 1; j++) {
                p[i][j] = (((pn[i][j+1] + pn[i][j-1]) * dy * dy +
                            (pn[i+1][j] + pn[i-1][j]) * dx * dx -
                            b[i][j] * dx * dx * dy * dy) /
                           (2.0 * (dx * dx + dy * dy)));
            }
        }

        // Enforce boundary conditions
        for (int i = 0; i < ny; i++) p[i][0] = p[i][1];
        for (int i = 0; i < ny; i++) p[i][nx-1] = p[i][nx-2];
        for (int j = 0; j < nx; j++) p[0][j] = p[1][j];
        for (int j = 0; j < nx; j++) p[ny-1][j] = 0;
    }
}

// Main simulation loop
void simulate() {
    vector<vector<double>> u(ny, vector<double>(nx, 0.0));
    vector<vector<double>> v(ny, vector<double>(nx, 0.0));
    vector<vector<double>> p(ny, vector<double>(nx, 0.0));
    vector<vector<double>> b(ny, vector<double>(nx, 0.0));

    for (int n = 0; n < nt; n++) {
        vector<vector<double>> un = u;
        vector<vector<double>> vn = v;

        #pragma omp parallel for collapse(2)
        for (int i = 1; i < ny - 1; i++) {
            for (int j = 1; j < nx - 1; j++) {
                b[i][j] = rho * (1.0 / dt * ((un[i][j+1] - un[i][j-1]) / (2 * dx) +
                                             (vn[i+1][j] - vn[i-1][j]) / (2 * dy))
                             - pow((un[i][j+1] - un[i][j-1]) / (2 * dx), 2)
                             - 2.0 * ((un[i+1][j] - un[i-1][j]) / (2 * dy) *
                                      (vn[i][j+1] - vn[i][j-1]) / (2 * dx))
                             - pow((vn[i+1][j] - vn[i-1][j]) / (2 * dy), 2));
            }
        }

        pressure_poisson(p, b);

        #pragma omp parallel for collapse(2)
        for (int i = 1; i < ny - 1; i++) {
            for (int j = 1; j < nx - 1; j++) {
                u[i][j] = un[i][j] - un[i][j] * dt / dx * (un[i][j] - un[i][j-1])
                                   - vn[i][j] * dt / dy * (un[i][j] - un[i-1][j])
                                   - dt / (2 * rho * dx) * (p[i][j+1] - p[i][j-1])
                                   + nu * dt * ((un[i][j+1] - 2 * un[i][j] + un[i][j-1]) / (dx * dx)
                                               + (un[i+1][j] - 2 * un[i][j] + un[i-1][j]) / (dy * dy));
                v[i][j] = vn[i][j] - un[i][j] * dt / dx * (vn[i][j] - vn[i][j-1])
                                   - vn[i][j] * dt / dy * (vn[i][j] - vn[i-1][j])
                                   - dt / (2 * rho * dy) * (p[i+1][j] - p[i-1][j])
                                   + nu * dt * ((vn[i][j+1] - 2 * vn[i][j] + vn[i][j-1]) / (dx * dx)
                                               + (vn[i+1][j] - 2 * vn[i][j] + vn[i-1][j]) / (dy * dy));
            }
        }

        // Apply boundary conditions
        for (int j = 0; j < nx; j++) u[0][j] = 1.0;  // Lid velocity
        for (int j = 0; j < nx; j++) u[ny-1][j] = 0;
        for (int i = 0; i < ny; i++) u[i][0] = u[i][nx-1] = 0;
        for (int i = 0; i < ny; i++) v[i][0] = v[i][nx-1] = 0;
        for (int j = 0; j < nx; j++) v[0][j] = v[ny-1][j] = 0;
    }

    // Write results to CSV for visualization
    ofstream file("velocity_field.csv");
    file << "X,Y,U,V" << endl;
    for (int i = 0; i < ny; i++) {
        for (int j = 0; j < nx; j++) {
            file << j * dx << "," << i * dy << "," << u[i][j] << "," << v[i][j] << endl;
        }
    }
    file.close();
    cout << "Simulation complete. Data saved to velocity_field.csv" << endl;
}

int main() {
    simulate();
    return 0;
}

