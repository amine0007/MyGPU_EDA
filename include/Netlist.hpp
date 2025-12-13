#pragma once

#define GATE_AND 0
#define GATE_OR  1
#define GATE_XOR 2
#define GATE_NOT 3
#define GATE_DFF 4  // <--- NOUVEAU : La Mémoire

struct Gate {
    int type;
    int input1_id;
    int input2_id;
    int output_id;
};