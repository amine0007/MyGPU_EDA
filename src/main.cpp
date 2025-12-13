#include <SFML/Graphics.hpp>
#include <iostream>
#include <vector>
#include <fstream>  // Pour lire les fichiers
#include <sstream>  // Pour découper les textes
#include "Netlist.hpp"

const int NB_SAMPLES = 1000;

// Signature mise à jour : on envoie le circuit (Gates) au GPU
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
            std::cout << "[PARSER] Ajout porte " << typeStr << " (In:" << in1 << "," << in2 << " -> Out:" << out << ")" << std::endl;
        }
    }
    return gates;
}

int main() {
    sf::RenderWindow window(sf::VideoMode(1200, 400), "MyGPU_EDA - Dynamic Loader");
    window.setFramerateLimit(60);

    std::vector<int> resultats(NB_SAMPLES);
    
    // 1. CHARGEMENT DU FICHIER
    std::cout << "--- Chargement du circuit ---" << std::endl;
    std::vector<Gate> monCircuit = loadCircuit("circuit.txt");
    std::cout << "Total portes chargees : " << monCircuit.size() << std::endl;

    if (monCircuit.empty()) {
        std::cout << "ATTENTION: Circuit vide ou fichier introuvable !" << std::endl;
    }

    // 2. SIMULATION
    #ifdef USE_CUDA
        std::cout << "Lancement Simulation GPU..." << std::endl;
        // On passe le vecteur de portes au GPU
        launchGPUCalculation(monCircuit.data(), monCircuit.size(), resultats.data(), NB_SAMPLES);
        
        // Debug Console
        std::cout << "Resultats (Sample): ";
        for(int i=0; i<10; i++) std::cout << resultats[i] << " ";
        std::cout << std::endl;
    #else
        for(int i=0; i<NB_SAMPLES; i++) resultats[i] = 0; 
    #endif

    // 3. VISUALISATION (Code identique à avant)
    sf::VertexArray waveform(sf::LineStrip, NB_SAMPLES * 2);
    float zoom = 50.0f;
    float offsetX = 0.0f;
    float yBase = 250.0f;
    float hauteur = 100.0f;

    while (window.isOpen()) {
        sf::Event event;
        while (window.pollEvent(event)) {
            if (event.type == sf::Event::Closed) window.close();
            
            // --- GESTION CLAVIER ---
            if (event.type == sf::Event::KeyPressed) {
                // Zoom : Flèches Haut/Bas
                if (event.key.code == sf::Keyboard::Up) zoom *= 1.1f;
                if (event.key.code == sf::Keyboard::Down) zoom *= 0.9f;
                
                // Déplacement : Flèches Gauche/Droite
                if (event.key.code == sf::Keyboard::Left) offsetX += 50.0f;
                if (event.key.code == sf::Keyboard::Right) offsetX -= 50.0f;
            }
        }

        window.clear(sf::Color::Black);

        // Recalcul des positions à chaque image
        for (int i = 0; i < NB_SAMPLES; i++) {
            int val = resultats[i];
            float x = i * zoom + offsetX; // On ajoute le décalage (scroll)
            float y = yBase - (val * hauteur);

            // Optimisation : Ne pas dessiner ce qui est hors écran
            if (x < -50 || x > 1250) continue;

            waveform[i * 2].position = sf::Vector2f(x, y);
            waveform[i * 2 + 1].position = sf::Vector2f(x + zoom, y);
        }

        window.draw(waveform);
        window.display();
    }

    for (int i = 0; i < NB_SAMPLES; i++) {
        int val = resultats[i];
        float x = i * zoom;
        float y = yBase - (val * hauteur);
        waveform[i * 2].position = sf::Vector2f(x, y); waveform[i * 2].color = sf::Color::Green;
        waveform[i * 2 + 1].position = sf::Vector2f(x + zoom, y); waveform[i * 2 + 1].color = sf::Color::Green;
    }

    while (window.isOpen()) {
        sf::Event event;
        while (window.pollEvent(event)) {
            if (event.type == sf::Event::Closed) window.close();
        }
        window.clear(sf::Color::Black);
        window.draw(waveform);
        window.display();
    }
    return 0;
}