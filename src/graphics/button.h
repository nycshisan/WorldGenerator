//
// Created by Nycshisan on 2018/3/8.
//

#ifndef WORLDGENERATOR_BUTTON_H
#define WORLDGENERATOR_BUTTON_H

#include <vector>
#include <utility>
#include <functional>

#include "SFML/Graphics.hpp"

class Button : public sf::RectangleShape {
public:
    void update() {

    }


    static std::vector<std::pair<std::string, std::function<void(void)>>> PanelButtonMaterials;
};

#endif //WORLDGENERATOR_BUTTON_H
