//
// Created by Nycshisan on 2018/3/8.
//

#include "button.h"

namespace wg {

    void Button::setFont(const sf::Font &font) {
        _text.setFont(font);
    }

    void Button::setLabel(const std::string &label) {
        _text.setString(label);
    }

    void Button::setResponder(const ButtonResponder &responder) {
        _responder = responder;
    }

    void Button::setSize(const sf::Vector2f &size) {
        sf::RectangleShape::setSize(size);
        setOutlineThickness(float(size.y * 0.1));
        _text.setCharacterSize(unsigned(size.y * 0.8));
    }

    void Button::setPosition(const sf::Vector2f &position) {
        sf::RectangleShape::setPosition(position);
        float textWidth = _text.getLocalBounds().width;
        float textOffset = (getSize().x - textWidth) / 2;
        _text.setPosition(position.x + textOffset, position.y);
    }

    void Button::setColor(const sf::Color &color) {
        _color = color;
        _invColor.r = sf::Uint8(255) - color.r;
        _invColor.g = sf::Uint8(255) - color.g;
        _invColor.b = sf::Uint8(255) - color.b;
        _invColor.a = color.a;

        setFillColor(_invColor);
        setOutlineColor(_color);
        _text.setFillColor(_color);
    }

    void Button::drawTo(sf::RenderWindow &window) {
        window.draw(*this);
        window.draw(_text);
    }

    void Button::hover(const sf::Window &window) {
        auto mousePos = sf::Mouse::getPosition(window);
        if (_checkPosInButton(mousePos.x, mousePos.y)) {
            setFillColor(_color);
            _text.setFillColor(_invColor);
        } else {
            setColor(_color);
        }
    }

    void Button::respond(Window &window, int x, int y) {
        if (_checkPosInButton(x, y)) {
            _responder(window);
        }
    }

    bool Button::_checkPosInButton(int x, int y) {
        auto bounds = getGlobalBounds();
        return bounds.left < x &&
               bounds.left + bounds.width > x &&
               bounds.top < y &&
               bounds.top + bounds.height > y;
    }

}
