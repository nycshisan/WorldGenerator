//
// Created by Nycshisan on 2018/3/8.
//

#include "window.h"

Window::Window(unsigned int width, unsigned int height, unsigned int barHeight) : sf::RenderWindow(sf::VideoMode(width, height + 2 * barHeight), "") {
    setTitle(_defaultTitle);
    _width = width;
    _height = height;
    _barHeight = barHeight;

    const Configure &conf = Configure::SharedInstance();

    // Initialize bar separators
    int barSeparatorHeight = conf["ui"]["barSeparatorHeight"].GetInt();
    _barSeparator.setSize(sf::Vector2f(_width, barSeparatorHeight));

    // Initialize hint label
    std::string font_fn(conf["ui"]["font"].GetString());
    _font.loadFromFile(font_fn);

    _hintLabel.setFont(_font);

    auto hintCharacterSize = (unsigned int)((_barHeight - barSeparatorHeight) * 0.8);
    _hintLabel.setCharacterSize(hintCharacterSize);

    _hintLabel.setFillColor(sf::Color::White);

    _hintLabel.setPosition(0, _height);

    _hintLabelContent = "Ready!";

    // Initialize panel
}

void Window::play() {
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

void Window::_displayBar() {
    // Display separators
    _barSeparator.setPosition(0, _height);
    draw(_barSeparator);
    _barSeparator.setPosition(0, _height + _barHeight);
    draw(_barSeparator);

    // Display hint label
    _hintLabel.setString(_hintLabelContent);
    draw(_hintLabel);

    // Display panel
    for (auto &button : _buttons) {
        button.update();
        draw(button);
    }
}
