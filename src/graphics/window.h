//
// Created by Nycshisan on 2018/3/8.
//

#ifndef WORLDGENERATOR_WINDOW_H
#define WORLDGENERATOR_WINDOW_H

#include <vector>

#include "SFML/Window.hpp"
#include "SFML/Graphics.hpp"

#include "../conf/conf.h"
#include "../misc/log.h"
#include "button.h"

class Window : sf::RenderWindow {
    std::string _defaultTitle = "World Generator";
    unsigned int _width, _height, _barHeight;

    sf::RectangleShape _barSeparator;

    sf::Font _font;

    sf::Text _hintLabel;
    std::string _hintLabelContent;

    std::vector<Button> _buttons;

    void _displayBar();

public:
    Window(unsigned int width, unsigned int height, unsigned int barHeight);

    void play();
};

#endif //WORLDGENERATOR_WINDOW_H
