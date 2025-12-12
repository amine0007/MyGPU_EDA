#include <cuda_runtime.h>
#include <iostream>
#include "Netlist.hpp" // On inclut notre nouvelle structure

__global__ void simulateCircuit(Gate* gates, int* signals, int nbGates) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < nbGates) {
        Gate g = gates[idx];
        
        // Lecture des valeurs d'entrée depuis le tableau global "signals"
        int val1 = signals[g.input1_id];
        int val2 = signals[g.input2_id];
        int result = 0;

        // La logique câblée
        switch(g.type) {
            case GATE_AND: result = val1 & val2; break;
            case GATE_OR:  result = val1 | val2; break;
            case GATE_XOR: result = val1 ^ val2; break;
            case GATE_NOT: result = !val1;       break; // !0 = 1, !1 = 0
        }

        // Écriture du résultat
        signals[g.output_id] = result;
    }
}

extern "C" void launchGPUCalculation(int* host_waveform_out, int nb_cycles) {
    // --- 1. DÉFINITION DU CIRCUIT (On crée 3 portes manuellement pour tester) ---
    // Fil 0 et 1 = Entrées. Fil 2 = Sortie AND. Fil 3 = Sortie OR. Fil 4 = Sortie XOR.
    int numGates = 3;
    size_t gateSize = numGates * sizeof(Gate);
    
    Gate h_gates[3];
    // Porte 0 : AND (Fil 0 & Fil 1 -> Fil 2)
    h_gates[0] = {GATE_AND, 0, 1, 2}; 
    // Porte 1 : OR  (Fil 0 | Fil 1 -> Fil 3)
    h_gates[1] = {GATE_OR,  0, 1, 3}; 
    // Porte 2 : XOR (Fil 0 ^ Fil 1 -> Fil 4)
    h_gates[2] = {GATE_XOR, 0, 1, 4};

    // --- 2. PRÉPARATION MÉMOIRE ---
    int numSignals = 5; // On a 5 fils au total
    int* d_signals;
    Gate* d_gates;
    
    cudaMalloc(&d_signals, numSignals * sizeof(int));
    cudaMalloc(&d_gates, gateSize);
    
    // On envoie la liste des portes au GPU (une seule fois, ça ne change pas)
    cudaMemcpy(d_gates, h_gates, gateSize, cudaMemcpyHostToDevice);

    // --- 3. BOUCLE TEMPORELLE (Simulation cycle par cycle) ---
    // Pour chaque instant t, on met à jour les entrées et on calcule
    for (int t = 0; t < nb_cycles; t++) {
        
        // A. On génère des signaux d'entrée (Stimuli)
        // Fil 0 : Horloge rapide (0, 1, 0, 1...)
        // Fil 1 : Horloge lente  (0, 0, 1, 1...)
        int inputs[2];
        inputs[0] = t % 2;
        inputs[1] = (t / 3) % 2; 

        // On envoie juste les entrées au GPU
        cudaMemcpy(d_signals, inputs, 2 * sizeof(int), cudaMemcpyHostToDevice);

        // B. On lance le calcul des portes
        simulateCircuit<<<1, 256>>>(d_gates, d_signals, numGates);
        cudaDeviceSynchronize();

        // C. On récupère le résultat du Fil 4 (XOR) pour l'afficher
        int val_sortie;
        cudaMemcpy(&val_sortie, &d_signals[4], sizeof(int), cudaMemcpyDeviceToHost);
        
        // On stocke pour l'affichage
        host_waveform_out[t] = val_sortie;
    }

    cudaFree(d_signals);
    cudaFree(d_gates);
}