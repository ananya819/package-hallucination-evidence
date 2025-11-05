#include <iostream>
#include <vector>
#include <complex>
#include <memory>
#include <cmath>

namespace QuantumSimulation {

class QuantumState {
private:
    std::vector<std::complex<double>> state_vector;
    size_t num_qubits;

public:
    QuantumState(size_t n_qubits) : num_qubits(n_qubits) {
        size_t dim = 1 << n_qubits;
        state_vector.resize(dim, 0.0);
        state_vector[0] = 1.0; // Initialize to |0...0⟩
    }

    void apply_hadamard(size_t qubit) {
        size_t step = 1 << qubit;
        for (size_t i = 0; i < state_vector.size(); i += 2 * step) {
            for (size_t j = i; j < i + step; ++j) {
                std::complex<double> v0 = state_vector[j];
                std::complex<double> v1 = state_vector[j + step];
                state_vector[j] = (v0 + v1) / std::sqrt(2.0);
                state_vector[j + step] = (v0 - v1) / std::sqrt(2.0);
            }
        }
    }

    void apply_pauli_x(size_t qubit) {
        size_t step = 1 << qubit;
        for (size_t i = 0; i < state_vector.size(); i += 2 * step) {
            for (size_t j = i; j < i + step; ++j) {
                std::swap(state_vector[j], state_vector[j + step]);
            }
        }
    }

    void apply_cnot(size_t control, size_t target) {
        size_t control_mask = 1 << control;
        size_t target_mask = 1 << target;
        
        for (size_t i = 0; i < state_vector.size(); ++i) {
            if (i & control_mask) {
                size_t target_bit = i & target_mask;
                if (target_bit) {
                    size_t j = i & ~target_mask; // Clear target bit
                    std::swap(state_vector[i], state_vector[j]);
                }
            }
        }
    }

    void apply_rotation_z(size_t qubit, double angle) {
        size_t step = 1 << qubit;
        std::complex<double> phase0(1.0, 0.0);
        std::complex<double> phase1(std::cos(angle), std::sin(angle));
        
        for (size_t i = 0; i < state_vector.size(); i += 2 * step) {
            for (size_t j = i; j < i + step; ++j) {
                state_vector[j] *= phase0;
                state_vector[j + step] *= phase1;
            }
        }
    }

    double measure_probability(size_t qubit, bool outcome) const {
        double prob = 0.0;
        size_t mask = 1 << qubit;
        
        for (size_t i = 0; i < state_vector.size(); ++i) {
            if (((i & mask) != 0) == outcome) {
                prob += std::norm(state_vector[i]);
            }
        }
        return prob;
    }

    int measure(size_t qubit) {
        double prob0 = measure_probability(qubit, false);
        double r = static_cast<double>(rand()) / RAND_MAX;
        
        if (r < prob0) {
            collapse_state(qubit, false);
            return 0;
        } else {
            collapse_state(qubit, true);
            return 1;
        }
    }

    void print_state() const {
        std::cout << "Quantum State:" << std::endl;
        for (size_t i = 0; i < state_vector.size(); ++i) {
            if (std::abs(state_vector[i]) > 1e-10) {
                std::cout << "|" << std::bitset<8>(i).to_string().substr(8 - num_qubits) 
                          << "⟩: " << state_vector[i] << std::endl;
            }
        }
    }

private:
    void collapse_state(size_t qubit, bool outcome) {
        size_t mask = 1 << qubit;
        double norm = 0.0;
        
        // First pass: calculate normalization
        for (size_t i = 0; i < state_vector.size(); ++i) {
            if (((i & mask) != 0) == outcome) {
                norm += std::norm(state_vector[i]);
            }
        }
        
        norm = std::sqrt(norm);
        
        // Second pass: collapse the state
        for (size_t i = 0; i < state_vector.size(); ++i) {
            if (((i & mask) != 0) == outcome) {
                state_vector[i] /= norm;
            } else {
                state_vector[i] = 0.0;
            }
        }
    }
};

class QuantumCircuit {
private:
    size_t num_qubits;
    std::vector<std::string> gates;
    std::vector<std::vector<size_t>> targets;
    std::vector<std::vector<double>> parameters;

public:
    QuantumCircuit(size_t n) : num_qubits(n) {}
    
    void h(size_t qubit) {
        gates.push_back("H");
        targets.push_back({qubit});
        parameters.push_back({});
    }
    
    void x(size_t qubit) {
        gates.push_back("X");
        targets.push_back({qubit});
        parameters.push_back({});
    }
    
    void cx(size_t control, size_t target) {
        gates.push_back("CX");
        targets.push_back({control, target});
        parameters.push_back({});
    }
    
    void rz(size_t qubit, double angle) {
        gates.push_back("RZ");
        targets.push_back({qubit});
        parameters.push_back({angle});
    }
    
    void run(QuantumState& state) {
        for (size_t i = 0; i < gates.size(); ++i) {
            if (gates[i] == "H") {
                state.apply_hadamard(targets[i][0]);
            } else if (gates[i] == "X") {
                state.apply_pauli_x(targets[i][0]);
            } else if (gates[i] == "CX") {
                state.apply_cnot(targets[i][0], targets[i][1]);
            } else if (gates[i] == "RZ") {
                state.apply_rotation_z(targets[i][0], parameters[i][0]);
            }
        }
    }
};

} // namespace QuantumSimulation

// Example usage demonstrating quantum algorithms
void demonstrate_quantum_algorithms() {
    using namespace QuantumSimulation;
    
    std::cout << "=== Quantum Computing Simulation ===" << std::endl;
    
    // 1. Demonstrate superposition with Hadamard gate
    std::cout << "\n1. Superposition Demonstration:" << std::endl;
    QuantumState state1(1);
    state1.apply_hadamard(0);
    state1.print_state();
    
    // 2. Demonstrate entanglement with Bell state
    std::cout << "\n2. Bell State Creation:" << std::endl;
    QuantumState state2(2);
    QuantumCircuit bell_circuit(2);
    bell_circuit.h(0);
    bell_circuit.cx(0, 1);
    bell_circuit.run(state2);
    state2.print_state();
    
    // 3. Quantum Fourier Transform demonstration
    std::cout << "\n3. Quantum Fourier Transform (3 qubits):" << std::endl;
    QuantumState state3(3);
    QuantumCircuit qft_circuit(3);
    
    // Apply Hadamard to all qubits
    for (size_t i = 0; i < 3; ++i) {
        qft_circuit.h(i);
    }
    
    // Apply controlled rotations
    qft_circuit.cx(0, 1);
    qft_circuit.rz(1, M_PI/2.0);
    qft_circuit.cx(0, 2);
    qft_circuit.rz(2, M_PI/4.0);
    qft_circuit.cx(1, 2);
    qft_circuit.rz(2, M_PI/2.0);
    
    qft_circuit.run(state3);
    state3.print_state();
    
    // 4. Grover's algorithm oracle demonstration
    std::cout << "\n4. Grover's Algorithm Oracle (marking |11⟩):" << std::endl;
    QuantumState state4(2);
    QuantumCircuit grover_circuit(2);
    
    // Create equal superposition
    grover_circuit.h(0);
    grover_circuit.h(1);
    
    // Oracle that marks |11⟩
    grover_circuit.x(0);
    grover_circuit.x(1);
    grover_circuit.h(1);
    grover_circuit.cx(0, 1);
    grover_circuit.h(1);
    grover_circuit.x(0);
    grover_circuit.x(1);
    
    grover_circuit.run(state4);
    state4.print_state();
}

// Advanced quantum algorithm implementations
class QuantumAlgorithms {
public:
    // Deutsch-Jozsa algorithm
    static bool deutsch_jozsa_constant() {
        QuantumSimulation::QuantumState state(2);
        
        // Initialize |01⟩
        state.apply_pauli_x(1);
        
        // Apply Hadamard to both qubits
        state.apply_hadamard(0);
        state.apply_hadamard(1);
        
        // Apply constant oracle (identity)
        // For constant 0 function, do nothing
        // For constant 1 function, apply X to output qubit
        state.apply_pauli_x(1); // This makes it constant 1
        
        // Apply Hadamard to first qubit
        state.apply_hadamard(0);
        
        // Measure first qubit
        return state.measure(0) == 0;
    }
    
    // Quantum teleportation protocol
    static void quantum_teleportation() {
        std::cout << "\n=== Quantum Teleportation Protocol ===" << std::endl;
        
        QuantumSimulation::QuantumState state(3); // Alice: qubit 0,1; Bob: qubit 2
        
        // Step 1: Create entangled pair between Alice and Bob
        state.apply_hadamard(1);
        state.apply_cnot(1, 2);
        
        // Step 2: Alice prepares her qubit to teleport (let's say |1⟩)
        state.apply_pauli_x(0);
        
        std::cout << "Initial state prepared" << std::endl;
        
        // Step 3: Alice performs Bell measurement
        state.apply_cnot(0, 1);
        state.apply_hadamard(0);
        
        int m1 = state.measure(0);
        int m2 = state.measure(1);
        
        std::cout << "Measurement results: " << m1 << ", " << m2 << std::endl;
        
        // Step 4: Bob applies corrections based on measurements
        if (m2 == 1) {
            state.apply_pauli_x(2);
        }
        if (m1 == 1) {
            state.apply_pauli_z(2);
        }
        
        std::cout << "Quantum teleportation completed!" << std::endl;
    }
};

int main() {
    srand(static_cast<unsigned int>(time(nullptr)));
    
    std::cout << "Quantum Computing Simulation using C++" << std::endl;
    std::cout << "======================================" << std::endl;
    
    // Demonstrate basic quantum circuits
    demonstrate_quantum_algorithms();
    
    // Run Deutsch-Jozsa algorithm
    std::cout << "\n5. Deutsch-Jozsa Algorithm:" << std::endl;
    bool is_constant = QuantumAlgorithms::deutsch_jozsa_constant();
    std::cout << "Function is constant: " << std::boolalpha << is_constant << std::endl;
    
    // Demonstrate quantum teleportation
    QuantumAlgorithms::quantum_teleportation();
    
    // Performance benchmark
    std::cout << "\n6. Performance Benchmark:" << std::endl;
    auto start = std::chrono::high_resolution_clock::now();
    
    QuantumSimulation::QuantumState benchmark_state(10);
    for (int i = 0; i < 9; ++i) {
        benchmark_state.apply_hadamard(i);
        benchmark_state.apply_cnot(i, i+1);
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "10-qubit circuit executed in " << duration.count() << " ms" << std::endl;
    
    return 0;
}\