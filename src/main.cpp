#include <SFML/Graphics.hpp>
#include <iostream>
#include <vector>

const int NB_SAMPLES = 1000; // On affiche les 1000 premiers points

// Signature mise à jour
extern "C" void launchGPUCalculation(int* host_results, int N);

int main() {
    sf::RenderWindow window(sf::VideoMode(1200, 400), "MyGPU_EDA - Visualizer");
    window.setFramerateLimit(60);

    // 1. Préparer le tableau de réception
    std::vector<int> resultats(NB_SAMPLES);

    // 2. Lancer la simulation (Si CUDA dispo)
    #ifdef USE_CUDA
        std::cout << "Simulation GPU en cours..." << std::endl;
        launchGPUCalculation(resultats.data(), NB_SAMPLES);
    #else
        // Mode dégradé pour Codespaces (juste pour tester la compile)
        for(int i=0; i<NB_SAMPLES; i++) resultats[i] = i%2; 
    #endif

    // 3. Créer la forme géométrique (LineStrip = Ligne continue)
    // On multiplie par 2 car pour faire un carré, il faut 2 points par valeur
    sf::VertexArray waveform(sf::LineStrip, NB_SAMPLES * 2);

    float zoom = 10.0f; // 10 pixels par unité de temps
    float yBase = 200.0f;
    float hauteur = 50.0f;

    for (int i = 0; i < NB_SAMPLES; i++) {
        int val = resultats[i];
        float x = i * zoom;
        float y = yBase - (val * hauteur);

        // Point A (Début du cycle)
        waveform[i * 2].position = sf::Vector2f(x, y);
        waveform[i * 2].color = sf::Color::Green;

        // Point B (Fin du cycle, pour faire le plat)
        waveform[i * 2 + 1].position = sf::Vector2f(x + zoom, y);
        waveform[i * 2 + 1].color = sf::Color::Green;
        
        // Note: Pour faire des vrais carrés parfaits verticaux, c'est un peu plus complexe,
        // mais commençons par ça, ça fera une belle ligne brisée.
    }

    while (window.isOpen()) {
        sf::Event event;
        while (window.pollEvent(event)) {
            if (event.type == sf::Event::Closed)
                window.close();
        }

        window.clear(sf::Color::Black);
        
        // Grille de fond (Optionnel, juste une ligne milieu)
        sf::VertexArray axe(sf::Lines, 2);
        axe[0].position = sf::Vector2f(0, yBase); axe[0].color = sf::Color(50,50,50);
        axe[1].position = sf::Vector2f(1200, yBase); axe[1].color = sf::Color(50,50,50);
        window.draw(axe);

        // Dessiner l'onde
        window.draw(waveform);

        window.display();
    }

    return 0;
}