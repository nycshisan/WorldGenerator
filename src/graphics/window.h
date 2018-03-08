//
// Created by Nycshisan on 2018/3/8.
//

#ifndef WORLDGENERATOR_WINDOW_H
#define WORLDGENERATOR_WINDOW_H

#include "SFML/Window.hpp"
#include "SFML/Graphics.hpp"

#include "../conf/conf.h"

class Window : sf::RenderWindow {
    std::string _defaultTitle = "World Generator";
    unsigned int _width, _height, _barHeight;

    sf::RectangleShape _barSeparator;

    sf::Text _hintLabel;

    void _displayBar() {
        // Display separators
        _barSeparator.setPosition(0, _height);
        draw(_barSeparator);
        _barSeparator.setPosition(0, _height + _barHeight);
        draw(_barSeparator);

        // Display hint label

    }

public:
    explicit Window(unsigned int width, unsigned int height, unsigned int barHeight) : sf::RenderWindow(sf::VideoMode(width, height + 2 * barHeight), "") {
        setTitle(_defaultTitle);
        _width = width;
        _height = height;
        _barHeight = barHeight;

        const Configure &conf = Configure::SharedInstance();

        // Initialize bar separators
        int barSeparatorHeight = conf["ui"]["barSeparatorHeight"].GetInt();
        _barSeparator.setSize(sf::Vector2f(_width, barSeparatorHeight));

        // Initialize hint label



        // Initialize panel
    }

    void play() {
        while (isOpen()) {
            sf::Event event = {};
            while (pollEvent(event)) {
                if (event.type == sf::Event::Closed) {
                    close();
                }
            }

            sf::Color backgroundColor(0, 0, 0);
            clear(backgroundColor);

            _displayBar();

            display();
        }
    }
};

#endif //WORLDGENERATOR_WINDOW_H
