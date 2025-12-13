#include <SFML/Graphics.hpp>
#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include "Netlist.hpp"

const int NB_SAMPLES = 1000;

// Structure pour stocker les infos d'une sonde pour l'affichage
struct ProbeData {
    int id;
    std::string name;
};

extern "C" void launchGPUCalculation(Gate* host_gates, int numGates, 
                                     int* host_probes, int numProbes, 
                                     int* host_results, int N);

// --- FONCTION PARSER AMÉLIORÉE (Lit les noms) ---
std::vector<Gate> loadCircuit(const std::string& filename, std::vector<ProbeData>& probes) {
    std::vector<Gate> gates;
    std::ifstream file(filename);
    probes.clear();
    
    if (!file.is_open()) {
        std::cerr << "ERREUR: Impossible d'ouvrir " << filename << std::endl;
        return gates;
    }

    std::string line;
    while (std::getline(file, line)) {
        if (line.empty()) continue;
        std::stringstream ss(line);
        std::string typeStr;
        ss >> typeStr;

        if (typeStr == "PROBE") {
            int probeId;
            ss >> probeId;
            std::string probeName;
            // Tente de lire le nom, sinon met un nom par défaut
            if (!(ss >> probeName)) {
                probeName = "Fil " + std::to_string(probeId);
            }
            probes.push_back({probeId, probeName});
            std::cout << "[PARSER] Sonde sur fil " << probeId << " (" << probeName << ")" << std::endl;
        }
        else {
            int in1, in2, out;
            ss >> in1 >> in2 >> out;
            int typeCode = -1;
            if (typeStr == "AND") typeCode = GATE_AND;
            else if (typeStr == "OR") typeCode = GATE_OR;
            else if (typeStr == "XOR") typeCode = GATE_XOR;
            else if (typeStr == "NOT") typeCode = GATE_NOT;
            else if (typeStr == "DFF") typeCode = GATE_DFF;
            if (typeCode != -1) gates.push_back({typeCode, in1, in2, out});
        }
    }
    return gates;
}

int main() {
    sf::RenderWindow window(sf::VideoMode(1200, 800), "MyGPU_EDA - Logic Analyzer Pro");
    window.setFramerateLimit(60);

    // 1. CHARGEMENT POLICE (Indispensable pour le texte)
    sf::Font font;
    if (!font.loadFromFile("arial.ttf")) {
        std::cerr << "ERREUR CRITIQUE : arial.ttf introuvable !" << std::endl;
        return -1;
    }

    // 2. CHARGEMENT CIRCUIT
    std::vector<ProbeData> displayProbes;
    std::vector<Gate> monCircuit = loadCircuit("circuit.txt", displayProbes);

    if (displayProbes.empty()) displayProbes.push_back({4, "Default Output"});

    // 3. PRÉPARATION GPU
    int nbProbes = displayProbes.size();
    // On extrait juste les IDs pour le GPU
    std::vector<int> gpuProbesIds;
    for(auto& p : displayProbes) gpuProbesIds.push_back(p.id);

    std::vector<int> resultats(NB_SAMPLES * nbProbes);

    #ifdef USE_CUDA
        launchGPUCalculation(monCircuit.data(), monCircuit.size(), 
                             gpuProbesIds.data(), nbProbes, 
                             resultats.data(), NB_SAMPLES);
    #else
        for(int t=0; t<NB_SAMPLES; t++) 
            for(int p=0; p<nbProbes; p++) resultats[t*nbProbes + p] = (t+p)%2;
    #endif

    // 4. PALETTE DE COULEURS (Style Oscilloscope)
    std::vector<sf::Color> palette = {
        sf::Color(0, 255, 255),   // Cyan (Entrée A)
        sf::Color(255, 0, 255),   // Magenta (Entrée B)
        sf::Color(255, 255, 0),   // Jaune (Entrée Cin)
        sf::Color(50, 255, 50),   // Vert Lime (Sortie Somme)
        sf::Color(255, 100, 100)  // Rouge Clair (Sortie Cout)
    };

    float zoom = 30.0f;
    float offsetX = 50.0f;
    float rowHeight = 120.0f;

    while (window.isOpen()) {
        sf::Event event;
        while (window.pollEvent(event)) {
            if (event.type == sf::Event::Closed) window.close();
        }

        if (sf::Keyboard::isKeyPressed(sf::Keyboard::Up))   zoom *= 1.02f;
        if (sf::Keyboard::isKeyPressed(sf::Keyboard::Down)) zoom *= 0.98f;
        if (sf::Keyboard::isKeyPressed(sf::Keyboard::Left))  offsetX += 10.0f;
        if (sf::Keyboard::isKeyPressed(sf::Keyboard::Right)) offsetX -= 10.0f;
        if (zoom < 1.0f) zoom = 1.0f; 

        window.clear(sf::Color(20, 20, 20));

        // --- BOUCLE D'AFFICHAGE ---
        for (int p = 0; p < nbProbes; p++) {
            float yBase = 100.0f + (p * rowHeight);
            // Choix de la couleur dans la palette (cycle si plus de 5 sondes)
            sf::Color channelColor = palette[p % palette.size()];

            // Ligne de référence
            sf::VertexArray refLine(sf::Lines, 2);
            refLine[0].position = sf::Vector2f(0, yBase); refLine[0].color = sf::Color(60,60,60);
            refLine[1].position = sf::Vector2f(1200, yBase); refLine[1].color = sf::Color(60,60,60);
            window.draw(refLine);

            // --- DESSIN DU LABEL (TEXTE) ---
            sf::Text label;
            label.setFont(font);
            label.setString(displayProbes[p].name); // Utilise le nom lu dans le fichier
            label.setCharacterSize(16);
            label.setFillColor(channelColor);       // Même couleur que le signal
            label.setPosition(10, yBase - 25);
            window.draw(label);

            // --- DESSIN DE LA COURBE ---
            sf::VertexArray waveform(sf::LineStrip);
            for (int t = 0; t < NB_SAMPLES; t++) {
                int val = resultats[t * nbProbes + p];
                float x = t * zoom + offsetX;
                float y = yBase - (val * 80.0f);

                if (x > -50 && x < 1250) {
                    // On applique la couleur du canal aux points
                    waveform.append(sf::Vertex(sf::Vector2f(x, y), channelColor));
                    waveform.append(sf::Vertex(sf::Vector2f(x + zoom, y), channelColor));
                }
            }
            window.draw(waveform);
        }
        window.display();
    }
    return 0;
}