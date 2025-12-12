#include <SFML/Graphics.hpp>
#include <iostream>

// Déclaration de la fonction externe (qui est dans le .cu)
extern "C" void launchGPUCalculation();

int main() {
    std::cout << "=== MyGPU_EDA Simulator ===" << std::endl;

    // Test de la simulation GPU
    #ifdef USE_CUDA
        launchGPUCalculation();
    #else
        std::cout << "Erreur: CUDA non active." << std::endl;
    #endif

    // La fenêtre SFML (On la garde pour plus tard)
    sf::RenderWindow window(sf::VideoMode(400, 200), "GPU Simulation Done");
    
    while (window.isOpen()) {
        sf::Event event;
        while (window.pollEvent(event)) {
            if (event.type == sf::Event::Closed)
                window.close();
        }
        window.clear();
        window.display();
    }

    return 0;
}