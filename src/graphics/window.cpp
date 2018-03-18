//
// Created by Nycshisan on 2018/3/8.
//

#include <string>

#include "window.h"

#include "../generator/generator.h"
#include "../conf/conf.h"

const static std::vector<std::pair<std::string, std::function<void(Window&)>>> _ButtonMaterials = { // NOLINT
        {"Next", NextButtonResponder},
        {"Redo", RedoButtonResponder},
        {"Save", SaveButtonResponder}
};

Window::Window(unsigned int width, unsigned int height, unsigned int barHeight) : sf::RenderWindow(sf::VideoMode(width, height + 2 * barHeight), "") {
    setTitle(_defaultTitle);
    setVerticalSyncEnabled(true);
    _width = width;
    _height = height;
    _barHeight = barHeight;

    const Configure &conf = CONF;

    _barSeparatorHeight = conf["ui"]["barSeparatorHeight"].GetUint();

    // Initialize bar separators
    _barSeparator.setSize(sf::Vector2f(_width, _barSeparatorHeight));

    // Initialize hint label
    int xOffset = _barHeight / 5;
    std::string font_fn(conf["ui"]["font"].GetString());
    _font.loadFromFile(font_fn);

    _hintLabel.setFont(_font);

    auto hintCharacterSize = (unsigned int)((_barHeight - _barSeparatorHeight) * 0.8);
    _hintLabel.setCharacterSize(hintCharacterSize);

    _hintLabel.setFillColor(sf::Color::White);

    _hintLabel.setPosition(xOffset, _height);

    _hintLabelContent = "Ready!";

    // Initialize panel
    auto buttonSize = sf::Vector2f(_barHeight * 3.0f, _barHeight * 0.6f);
    int buttonXOffset = xOffset;
    int buttonXInterval = (_width - (int)_ButtonMaterials.size() * (int)buttonSize.x - 2 * buttonXOffset) / ((int)_ButtonMaterials.size() - 1);
    for (auto &material : _ButtonMaterials) {
        Button button;
        button.setFont(_font);
        button.setLabel(material.first);
        button.setResponder(material.second);

        button.setColor(sf::Color::White);
        button.setSize(buttonSize);
        button.setPosition(sf::Vector2f(buttonXOffset, _height + _barHeight + _barSeparatorHeight / 2 + (_barHeight - buttonSize.y) / 2));
        buttonXOffset += buttonSize.x + buttonXInterval;

        _buttons.emplace_back(button);
    }
}

void Window::play() {
    NextButtonResponder(*this);
    NextButtonResponder(*this);
    while (isOpen()) {
        sf::Event event = {};
        while (pollEvent(event)) {
            if (event.type == sf::Event::Closed) {
                close();
            }
            if (event.type == sf::Event::MouseButtonReleased) {
                auto mouseEvent = event.mouseButton;
                if (mouseEvent.button == sf::Mouse::Button::Left) {
                    for (auto &button: _buttons) {
                        button.respond(*this, mouseEvent.x, mouseEvent.y);
                    }
                }
            }
        }

        _updateFPS();

        sf::Color backgroundColor(0, 0, 0);
        clear(backgroundColor);

        _displayBar();
        _displayMap();

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
        button.hover(*this);
        button.drawTo(*this);
    }
}

void Window::_displayMap() {
    Generator::SharedInstance().display(*this);
}

void Window::setHintLabel(const std::string &content) {
    _hintLabelContent = content;
}

sf::Vector2u Window::getMapSize() {
    return {_width, _height};
}

void Window::_updateFPS() {
    float interval = _clock.restart().asSeconds();
    float FPS = 1.0f / interval;
    if (_updateFPSCounter == _updateFPSFrameInterval - 1) {
        sprintf(_titleBuffer, "%s - FPS: %.1f", _defaultTitle, FPS);
        setTitle(_titleBuffer);
    }
    _updateFPSCounter = (_updateFPSCounter + 1) % _updateFPSFrameInterval;
}
