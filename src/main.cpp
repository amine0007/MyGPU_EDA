#include <SFML/Graphics.hpp>
#include <iostream>
#include <vector>
#include <Netlist.hpp>

// Nombre d'échantillons (cycles d'horloge) à simuler
const int NB_SAMPLES = 1000;

// Déclaration de la fonction CUDA (définie dans kernel.cu)
// extern "C" est crucial pour lier du C++ avec du CUDA
extern "C" void launchGPUCalculation(int* host_results, int N);

int main() {
    // Création de la fenêtre
    sf::RenderWindow window(sf::VideoMode(1200, 400), "MyGPU_EDA - Simulation XOR");
    window.setFramerateLimit(60);

    // Vecteur pour stocker les résultats (0 ou 1) qui viendront du GPU
    std::vector<int> resultats(NB_SAMPLES);

    std::cout << "=== MyGPU_EDA Simulator ===" << std::endl;

    // --- PARTIE SIMULATION ---
    #ifdef USE_CUDA
        std::cout << "[INFO] GPU NVIDIA detecte. Lancement du Kernel CUDA..." << std::endl;
        
        // Appel de la fonction magique (kernel.cu)
        launchGPUCalculation(resultats.data(), NB_SAMPLES);

        // --- VERIFICATION DEBUG (XOR) ---
        // On affiche les 8 premières valeurs dans la console pour vérifier la logique
        std::cout << "\n--- VERIFICATION LOGIQUE (XOR) ---" << std::endl;
        std::cout << "Entree A (Rapide): 0 1 0 1 0 1 0 1" << std::endl;
        std::cout << "Entree B (Lente) : 0 0 1 1 0 0 1 1" << std::endl;
        std::cout << "Attendu (XOR)    : 0 1 1 0 0 1 1 0" << std::endl;
        std::cout << "Recu du GPU      : ";
        
        for(int i=0; i<8; i++) {
            std::cout << resultats[i] << " ";
        }
        std::cout << "\n----------------------------------\n" << std::endl;
        // --------------------------------

    #else
        std::cout << "[WARN] Pas de CUDA detecte (Mode CPU/Codespaces). Simulation factice." << std::endl;
        // Remplissage bidon pour ne pas avoir un écran vide si tu testes sur Codespaces
        for(int i=0; i<NB_SAMPLES; i++) resultats[i] = (i % 4 == 1 || i % 4 == 2) ? 1 : 0; 
    #endif

    // --- PARTIE VISUALISATION (SFML) ---
    
    // On prépare les points géométriques
    // On utilise 2 points par échantillon pour dessiner des "marches" carrées
    sf::VertexArray waveform(sf::LineStrip, NB_SAMPLES * 2);

    float zoom = 50.0f;     // 50 pixels par unité de temps
    float yBase = 250.0f;   // Position Y du 0 (Bas de l'écran)
    float hauteur = 100.0f; // Hauteur du signal (Voltage)

    for (int i = 0; i < NB_SAMPLES; i++) {
        int val = resultats[i];
        
        // Calcul des coordonnées écran
        float x = i * zoom;
        float y = yBase - (val * hauteur); // On soustrait car Y=0 est en haut de l'écran

        // Point A (Début du cycle t)
        waveform[i * 2].position = sf::Vector2f(x, y);
        waveform[i * 2].color = sf::Color::Green;

        // Point B (Fin du cycle t, pour faire le plat du signal carré)
        waveform[i * 2 + 1].position = sf::Vector2f(x + zoom, y);
        waveform[i * 2 + 1].color = sf::Color::Green;
    }

    // --- BOUCLE D'AFFICHAGE ---
    while (window.isOpen()) {
        sf::Event event;
        while (window.pollEvent(event)) {
            if (event.type == sf::Event::Closed)
                window.close();
        }

        window.clear(sf::Color::Black);

        // 1. Dessiner une ligne grise pour l'axe 0
        sf::VertexArray axe(sf::Lines, 2);
        axe[0].position = sf::Vector2f(0, yBase);     axe[0].color = sf::Color(50, 50, 50);
        axe[1].position = sf::Vector2f(1200, yBase);  axe[1].color = sf::Color(50, 50, 50);
        window.draw(axe);

        // 2. Dessiner l'onde verte
        window.draw(waveform);

        window.display();
    }

    return 0;
}