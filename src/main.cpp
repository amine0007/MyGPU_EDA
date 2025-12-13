#include <SFML/Graphics.hpp>
#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include "Netlist.hpp"

const int NB_SAMPLES = 1000;

// Signature mise à jour pour accepter les sondes
extern "C" void launchGPUCalculation(Gate* host_gates, int numGates, 
                                     int* host_probes, int numProbes, 
                                     int* host_results, int N);

// --- FONCTION PARSER ---
// Retourne les portes ET remplit la liste des sondes trouvées
std::vector<Gate> loadCircuit(const std::string& filename, std::vector<int>& probes) {
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
        ss >> typeStr; // Lit le premier mot

        if (typeStr == "PROBE") {
            int probeId;
            ss >> probeId;
            probes.push_back(probeId);
            std::cout << "[PARSER] Ajout Sonde sur fil " << probeId << std::endl;
        }
        else {
            int in1, in2, out;
            ss >> in1 >> in2 >> out;
            int typeCode = -1;
            if (typeStr == "AND") typeCode = GATE_AND;
            else if (typeStr == "OR") typeCode = GATE_OR;
            else if (typeStr == "XOR") typeCode = GATE_XOR;
            else if (typeStr == "NOT") typeCode = GATE_NOT;
            
            if (typeCode != -1) gates.push_back({typeCode, in1, in2, out});
        }
    }
    return gates;
}

int main() {
    sf::RenderWindow window(sf::VideoMode(1200, 800), "MyGPU_EDA - Logic Analyzer");
    window.setFramerateLimit(60);

    std::vector<int> probes;
    std::vector<Gate> monCircuit = loadCircuit("circuit.txt", probes);

    // Si pas de sondes, on en met une par défaut (fil 4)
    if (probes.empty()) probes.push_back(4);

    int nbProbes = probes.size();
    // Le tableau de résultats doit contenir : NB_SAMPLES * Nombre de Sondes
    std::vector<int> resultats(NB_SAMPLES * nbProbes);

    #ifdef USE_CUDA
        launchGPUCalculation(monCircuit.data(), monCircuit.size(), 
                             probes.data(), nbProbes, 
                             resultats.data(), NB_SAMPLES);
    #else
        // Mode simulation bidon CPU
        for(int t=0; t<NB_SAMPLES; t++) {
            for(int p=0; p<nbProbes; p++) resultats[t*nbProbes + p] = (t+p)%2;
        }
    #endif

    // Interface
    float zoom = 30.0f;
    float offsetX = 50.0f;
    float rowHeight = 120.0f; // Espace vertical entre chaque courbe
    float signalHeight = 80.0f; // Hauteur du signal lui-même

    while (window.isOpen()) {
        sf::Event event;
        while (window.pollEvent(event)) {
            if (event.type == sf::Event::Closed) window.close();
        }

        // Contrôles
        if (sf::Keyboard::isKeyPressed(sf::Keyboard::Up))   zoom *= 1.02f;
        if (sf::Keyboard::isKeyPressed(sf::Keyboard::Down)) zoom *= 0.98f;
        if (sf::Keyboard::isKeyPressed(sf::Keyboard::Left))  offsetX += 10.0f;
        if (sf::Keyboard::isKeyPressed(sf::Keyboard::Right)) offsetX -= 10.0f;
        if (zoom < 1.0f) zoom = 1.0f; 

        window.clear(sf::Color(20, 20, 20)); // Gris très foncé

        // --- DESSIN DES PISTES ---
        for (int p = 0; p < nbProbes; p++) {
            // Position Y de base pour cette sonde
            float yBase = 100.0f + (p * rowHeight);
            
            // Ligne de référence grise
            sf::VertexArray refLine(sf::Lines, 2);
            refLine[0].position = sf::Vector2f(0, yBase); refLine[0].color = sf::Color(50,50,50);
            refLine[1].position = sf::Vector2f(1200, yBase); refLine[1].color = sf::Color(50,50,50);
            window.draw(refLine);

            // Petit texte (carré de couleur) pour identifier la ligne
            sf::RectangleShape badge(sf::Vector2f(20, 20));
            badge.setPosition(10, yBase - 10);
            badge.setFillColor(sf::Color::Yellow);
            window.draw(badge);

            // La Courbe
            sf::VertexArray waveform(sf::LineStrip); // On utilise LineStrip dynamique
            
            for (int t = 0; t < NB_SAMPLES; t++) {
                // Formule magique pour retrouver la bonne valeur dans le tableau 1D
                int val = resultats[t * nbProbes + p];
                
                float x = t * zoom + offsetX;
                float y = yBase - (val * signalHeight);

                if (x > -50 && x < 1250) {
                    waveform.append(sf::Vertex(sf::Vector2f(x, y), sf::Color::Green));
                    waveform.append(sf::Vertex(sf::Vector2f(x + zoom, y), sf::Color::Green));
                }
            }
            window.draw(waveform);
        }

        window.display();
    }
    return 0;
}