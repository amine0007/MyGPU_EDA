#include <cuda_runtime.h>
#include <iostream>
#include "Netlist.hpp"

__global__ void simulateCircuit(Gate* gates, int* signals, int nbGates) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < nbGates) {
        Gate g = gates[idx];
        int val1 = signals[g.input1_id];
        int val2 = signals[g.input2_id];
        int result = 0;
        switch(g.type) {
            case GATE_AND: result = val1 & val2; break;
            case GATE_OR:  result = val1 | val2; break;
            case GATE_XOR: result = val1 ^ val2; break;
            case GATE_NOT: result = !val1;       break;
        }
        signals[g.output_id] = result;
    }
}

// Nouvelle signature : on reçoit la liste des sondes (probes)
extern "C" void launchGPUCalculation(Gate* host_gates, int numGates, 
                                     int* host_probes, int numProbes, 
                                     int* host_waveform_out, int nb_cycles) {
    
    size_t gateSize = numGates * sizeof(Gate);
    int numSignals = 1024; 
    int* d_signals;
    Gate* d_gates;
    
    cudaMalloc(&d_signals, numSignals * sizeof(int));
    cudaMalloc(&d_gates, gateSize);
    cudaMemcpy(d_gates, host_gates, gateSize, cudaMemcpyHostToDevice);
    cudaMemset(d_signals, 0, numSignals * sizeof(int));

    // BOUCLE TEMPORELLE
    for (int t = 0; t < nb_cycles; t++) {
        
        // Génération Stimuli (Additionneur 3 bits : A, B, Cin)
        int inputs[3];
        inputs[0] = t % 2;           // Rapide
        inputs[1] = (t / 2) % 2;     // Moyen
        inputs[2] = (t / 4) % 2;     // Lent (Retenue entrée)

        cudaMemcpy(d_signals, inputs, 3 * sizeof(int), cudaMemcpyHostToDevice);

        int threads = 256;
        int blocks = (numGates + threads - 1) / threads;
        simulateCircuit<<<blocks, threads>>>(d_gates, d_signals, numGates);
        cudaDeviceSynchronize();

        // EXTRACTION DES SONDES (Multi-Canaux)
        for (int p = 0; p < numProbes; p++) {
            int val_sonde;
            // On va chercher la valeur du fil demandé par la sonde p
            int id_fil = host_probes[p];
            cudaMemcpy(&val_sonde, &d_signals[id_fil], sizeof(int), cudaMemcpyDeviceToHost);
            
            // On range ça dans le grand tableau de sortie
            // Structure : [Temps 0 - Sonde 0, Temps 0 - Sonde 1, ... Temps 1 - Sonde 0...]
            host_waveform_out[t * numProbes + p] = val_sonde;
        }
    }

    cudaFree(d_signals);
    cudaFree(d_gates);
}