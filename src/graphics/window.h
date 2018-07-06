//
// Created by Nycshisan on 2018/3/8.
//

#ifndef WORLDGENERATOR_WINDOW_H
#define WORLDGENERATOR_WINDOW_H

#include <vector>
#include <utility>
#include <functional>

#include "SFML/Window.hpp"
#include "SFML/Graphics.hpp"

#include "button.h"
#include "configWindow.h"
#include "../misc/misc.h"

namespace wg {

    class Generator;

    class MainWindow : public sf::RenderWindow {
        friend class ConfigWindow;

        static constexpr int _BaseBarHeight = 32;
        static constexpr int _BaseBarSeparatorHeight = 1;

        char _defaultTitle[20] = "World Generator";
        int _width, _height, _barHeight, _barSeparatorHeight;

        sf::RectangleShape _barSeparator;

        sf::Font _font;

        sf::Text _hintLabel;
        std::string _hintLabelContent;

        std::vector<Button> _buttons;

        char _titleBuffer[40];

        sf::Clock _clock;
        int _updateFPSCounter = 0, _updateFPSFrameInterval = 10;

        void _updateFPS();

        void _displayBar();

        void _displayMap();

        void _displayConfigWindow();

        ConfigWindow *configWindow = nullptr;

    public:
        MainWindow(int width, int height);

        void setHintLabel(const std::string &content);

        void play();

        void openConfigWindow(Generator *generator);
    };

}

#endif //WORLDGENERATOR_WINDOW_H
