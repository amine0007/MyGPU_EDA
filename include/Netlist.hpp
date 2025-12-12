#pragma once

// Types de portes supportées
#define GATE_AND 0
#define GATE_OR  1
#define GATE_XOR 2
#define GATE_NOT 3

// La structure d'une porte logique (tient sur 16 octets, parfait pour le GPU)
struct Gate {
    int type;      // Type de la porte (AND, OR...)
    int input1_id; // Index du fil d'entrée A
    int input2_id; // Index du fil d'entrée B (ignoré pour NOT)
    int output_id; // Index du fil de sortie
};