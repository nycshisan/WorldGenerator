//
// Created by Nycshisan on 2018/3/8.
//

#ifndef WORLDGENERATOR_BUTTON_H
#define WORLDGENERATOR_BUTTON_H

#include <functional>

#include "SFML/Graphics.hpp"
#include "SFML/Window.hpp"

namespace wg {

    class MainWindow;

    typedef std::function<void(MainWindow &)> ButtonResponder;

    class Button : public sf::RectangleShape {
        sf::Text _text;

        sf::Color _color, _invColor;

        ButtonResponder _responder;

        bool _checkPosInButton(int x, int y);

    public:
        void setFont(const sf::Font &font);

        void setLabel(const std::string &label);

        void setResponder(const ButtonResponder &responder);

        void setSize(const sf::Vector2f &size);

        void setPosition(const sf::Vector2f &position);

        void setColor(const sf::Color &color);

        void drawTo(sf::RenderWindow &window);

        void hover(const sf::Window &window);

        void respond(MainWindow &window, int x, int y);
    };

}

#endif //WORLDGENERATOR_BUTTON_H
