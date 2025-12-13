#include <SFML/Graphics.hpp>
#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include "Netlist.hpp"

const int NB_SAMPLES = 1000;

// Signature de la fonction CUDA
extern "C" void launchGPUCalculation(Gate* host_gates, int numGates, int* host_results, int N);

// --- FONCTION PARSER ---
std::vector<Gate> loadCircuit(const std::string& filename) {
    std::vector<Gate> gates;
    std::ifstream file(filename);
    
    if (!file.is_open()) {
        std::cerr << "ERREUR: Impossible d'ouvrir " << filename << std::endl;
        return gates;
    }

    std::string line;
    while (std::getline(file, line)) {
        if (line.empty()) continue;
        std::stringstream ss(line);
        std::string typeStr;
        int in1, in2, out;
        ss >> typeStr >> in1 >> in2 >> out;
        
        int typeCode = -1;
        if (typeStr == "AND") typeCode = GATE_AND;
        else if (typeStr == "OR") typeCode = GATE_OR;
        else if (typeStr == "XOR") typeCode = GATE_XOR;
        else if (typeStr == "NOT") typeCode = GATE_NOT;

        if (typeCode != -1) {
            gates.push_back({typeCode, in1, in2, out});
        }
    }
    return gates;
}

int main() {
    sf::RenderWindow window(sf::VideoMode(1200, 400), "MyGPU_EDA - Ultimate Simulator");
    window.setFramerateLimit(60);

    std::vector<int> resultats(NB_SAMPLES);
    
    // 1. CHARGEMENT
    std::vector<Gate> monCircuit = loadCircuit("circuit.txt");
    if (monCircuit.empty()) {
        // Circuit par défaut si fichier vide
        std::cout << "Fichier vide/absent. Chargement circuit test." << std::endl;
        monCircuit.push_back({GATE_XOR, 0, 1, 4}); 
    }

    // 2. SIMULATION
    #ifdef USE_CUDA
        launchGPUCalculation(monCircuit.data(), monCircuit.size(), resultats.data(), NB_SAMPLES);
    #else
        for(int i=0; i<NB_SAMPLES; i++) resultats[i] = i%2; 
    #endif

    // 3. VISUALISATION
    sf::VertexArray waveform(sf::LineStrip, NB_SAMPLES * 2);
    
    // Variables de navigation
    float zoom = 50.0f;
    float offsetX = 50.0f;
    float yBase = 250.0f;
    float hauteur = 100.0f;

    while (window.isOpen()) {
        sf::Event event;
        while (window.pollEvent(event)) {
            if (event.type == sf::Event::Closed) window.close();
        }

        // --- GESTION CLAVIER (FLUIDE) ---
        // On vérifie l'état des touches directement (hors de la boucle d'événements)
        
        // Zoom (Haut / Bas)
        if (sf::Keyboard::isKeyPressed(sf::Keyboard::Up))   zoom *= 1.02f; // Zoom In
        if (sf::Keyboard::isKeyPressed(sf::Keyboard::Down)) zoom *= 0.98f; // Zoom Out

        // Scroll (Gauche / Droite)
        if (sf::Keyboard::isKeyPressed(sf::Keyboard::Left))  offsetX += 10.0f;
        if (sf::Keyboard::isKeyPressed(sf::Keyboard::Right)) offsetX -= 10.0f;

        // Limites de sécurité pour éviter les bugs visuels
        if (zoom < 1.0f) zoom = 1.0f;       // Zoom minimum
        if (zoom > 500.0f) zoom = 500.0f;   // Zoom maximum

        window.clear(sf::Color::Black);

        // --- DESSIN INTELLIGENT ---
        int pointsDessines = 0;
        for (int i = 0; i < NB_SAMPLES; i++) {
            int val = resultats[i];
            
            // Calcul position
            float x = i * zoom + offsetX;
            float y = yBase - (val * hauteur);

            // OPTIMISATION : On ne met à jour que les points visibles à l'écran
            // (Entre -50px et la largeur de la fenêtre + 50px)
            if (x > -50 && x < 1250) {
                waveform[i * 2].position = sf::Vector2f(x, y);
                waveform[i * 2 + 1].position = sf::Vector2f(x + zoom, y);
                
                // Couleur verte Matrix
                waveform[i * 2].color = sf::Color(0, 255, 0);
                waveform[i * 2 + 1].color = sf::Color(0, 255, 0);
                pointsDessines++;
            } else {
                // Pour éviter les traits qui traversent tout l'écran, on "cache" les points hors champ
                waveform[i * 2].position = sf::Vector2f(x, y);
                waveform[i * 2].color = sf::Color::Transparent;
                waveform[i * 2 + 1].position = sf::Vector2f(x, y);
                waveform[i * 2 + 1].color = sf::Color::Transparent;
            }
        }

        // Axe horizontal (Référence)
        sf::VertexArray axe(sf::Lines, 2);
        axe[0].position = sf::Vector2f(0, yBase);     axe[0].color = sf::Color(100, 100, 100);
        axe[1].position = sf::Vector2f(1200, yBase);  axe[1].color = sf::Color(100, 100, 100);

        window.draw(axe);
        window.draw(waveform);
        window.display();
    }
    return 0;
}