//
// Created by Nycshisan on 2018/3/8.
//

#include <string>

#include "window.h"

#include "../conf/conf.h"
#include "../generator/generator.h"

namespace wg {

    const static std::vector<std::pair<std::string, std::function<void(MainWindow &)>>> _ButtonMaterials = { // NOLINT
            {"Next", Generator::NextButtonResponder},
            {"Redo", Generator::RedoButtonResponder},
            {"Undo", Generator::UndoButtonResponder},
            {"Save", Generator::SaveButtonResponder},
            {"Load", Generator::LoadButtonResponder},
            {"Config", Generator::ConfigButtonResponder}
    };

    MainWindow::MainWindow(int width, int height) : sf::RenderWindow(
            sf::VideoMode((unsigned int)width, (unsigned int)(height + _BaseBarHeight * CONF.getUIScale() * 2)), "") {
        setTitle(_defaultTitle);
        setVerticalSyncEnabled(true);
        _width = width;
        _height = height;
        float uiScale = CONF.getUIScale();
        _barHeight = int(_BaseBarHeight * uiScale);

        _barSeparatorHeight = int(_BaseBarSeparatorHeight * uiScale);

        // Initialize bar separators
        _barSeparator.setSize(sf::Vector2f(_width, _barSeparatorHeight));

        // Initialize hint label
        int xOffset = _barHeight / 5;
        std::string font_fn(CONF.getUIFontFilename());
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
        int buttonXInterval = (_width - (int) _ButtonMaterials.size() * (int) buttonSize.x - 2 * buttonXOffset) /
                              ((int) _ButtonMaterials.size() - 1);
        for (auto &material : _ButtonMaterials) {
            Button button;
            button.setFont(_font);
            button.setLabel(material.first);
            button.setResponder(material.second);

            button.setColor(sf::Color::White);
            button.setSize(buttonSize);
            button.setPosition(sf::Vector2f(buttonXOffset, _height + _barHeight + _barSeparatorHeight / 2.0f +
                                                           (_barHeight - buttonSize.y) / 2.0f));
            buttonXOffset += buttonSize.x + buttonXInterval;

            _buttons.emplace_back(button);
        }
    }

    void MainWindow::play() {
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
                if (event.type == sf::Event::KeyReleased) {
                    auto key = event.key;
                    if (key.code == sf::Keyboard::Key::Right) {
                        Generator::NextButtonResponder(*this);
                    }
                    if (key.code == sf::Keyboard::Key::Left) {
                        Generator::UndoButtonResponder(*this);
                    }
                    if (key.code == sf::Keyboard::Key::Up) {
                        Generator::LoadButtonResponder(*this);
                    }
                    if (key.code == sf::Keyboard::Key::Down) {
                        Generator::SaveButtonResponder(*this);
                    }
                    if (key.code == sf::Keyboard::Key::R) {
                        Generator::RedoButtonResponder(*this);
                    }
                }
            }

            _updateFPS();

            sf::Color backgroundColor(0, 0, 0);
            clear(backgroundColor);

            _displayMap();
            _displayBar();

            _displayConfigWindow();

            display();
        }
    }

    void MainWindow::_displayBar() {
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

    void MainWindow::_displayMap() {
        Generator::SharedInstance().display(*this);
    }

    void MainWindow::setHintLabel(const std::string &content) {
        _hintLabelContent = content;
    }

    void MainWindow::_updateFPS() {
        float interval = _clock.restart().asSeconds();
        float FPS = 1.0f / interval;
        if (_updateFPSCounter == _updateFPSFrameInterval - 1) {
            sprintf(_titleBuffer, "%s - FPS: %.1f", _defaultTitle, FPS);
            setTitle(_titleBuffer);
        }
        _updateFPSCounter = (_updateFPSCounter + 1) % _updateFPSFrameInterval;
    }

    void MainWindow::openConfigWindow(Generator *generator) {
        if (configWindow == nullptr) {
            configWindow = new ConfigWindow(generator);
            LOGOUT("Configuration window opened.");
        }
        configWindow->requestFocus();
    }

    void MainWindow::_displayConfigWindow() {
        if (configWindow != nullptr) {
            if (configWindow->isOpen()) {
                configWindow->play();
            } else {
                LOGOUT("Configuration window closed.");
                delete configWindow;
                configWindow = nullptr;
            }
        }
    }

}