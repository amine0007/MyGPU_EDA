#include <SFML/Graphics.hpp>
#include <iostream>

// Code spécifique CUDA (ne sera compilé que si CUDA est présent)
#ifdef USE_CUDA
    #include <cuda_runtime.h>
    void checkGPU() {
        int deviceCount;
        cudaGetDeviceCount(&deviceCount);
        std::cout << "Nombre de GPU NVIDIA detectes : " << deviceCount << std::endl;
    }
#else
    void checkGPU() {
        std::cout << "Pas de CUDA detecte (Mode CPU / Codespaces)" << std::endl;
    }
#endif

int main() {
    std::cout << "Demarrage de MyEDA..." << std::endl;
    checkGPU();

    sf::RenderWindow window(sf::VideoMode(800, 600), "MyGPU_EDA - Test");
    sf::CircleShape shape(100.f);
    shape.setFillColor(sf::Color::Green);

    while (window.isOpen()) {
        sf::Event event;
        while (window.pollEvent(event)) {
            if (event.type == sf::Event::Closed)
                window.close();
        }

        window.clear();
        window.draw(shape);
        window.display();
    }

    return 0;
}