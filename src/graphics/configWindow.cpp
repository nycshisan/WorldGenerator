//
// Created by Nycshisan on 2018/7/5.
//

#include "configWindow.h"

#include "../conf/conf.h"
#include "window.h"

sf::Font wg::ConfigUIWidget::_Font = sf::Font();
bool wg::ConfigUIWidget::_FontLoaded = false;

wg::ConfigWindow::ConfigWindow(wg::Generator *generator) : sf::RenderWindow(sf::VideoMode(
        (unsigned int)(_BaseConfigWindowWidth * CONF.getUIScale()),
        (unsigned int)(CONF.getMapHeight() + MainWindow::_BaseBarHeight * CONF.getUIScale() * 2) / 2), "Config") {
    this->_generator = generator;
    this->_lastState = Ready;
}

void wg::ConfigWindow::play() {
    if (_lastState != _generator->state) {
        _configWidgets.clear();
        int i = 0;
        for (auto &config: _generator->configs) {
            _configWidgets.emplace_back(config, i, *this); ++i;
        }
    }

    sf::Event event = {};
    while (pollEvent(event)) {
        if (event.type == sf::Event::Closed) {
            close();
        }
        if (event.type == sf::Event::MouseButtonPressed && event.mouseButton.button == sf::Mouse::Left) {
            for (auto &widget: _configWidgets) {
                widget.respond(event.mouseButton.x, event.mouseButton.y);
            }
        }
    }

    sf::Color backgroundColor(0, 0, 0);
    clear(backgroundColor);

    for (auto &widget: _configWidgets) {
        widget.display(*this);
    }

    display();
}

wg::ConfigUIWidget::ConfigUIWidget(const std::shared_ptr<GeneratorConfig> &config, int index, const ConfigWindow &window) {
    float uiScale = CONF.getUIScale();

    if (!_FontLoaded) {
        _Font.loadFromFile(CONF.getUIFontFilename());
        _FontLoaded = true;
    }
    _nameText.setFont(_Font);
    auto nameCharacterSize = static_cast<unsigned int>(_BaseValueRectHeight * 0.5f * uiScale);
    _nameText.setCharacterSize(nameCharacterSize);
    _nameText.setFillColor(sf::Color::White);
    _nameText.setString(config->name);
    _nameText.setOrigin(_nameText.getLocalBounds().width / 2.f, 0.f);
    sf::Vector2f nameTextPos;
    nameTextPos.x = float(window.getSize().x) / 2;
    nameTextPos.y =_BaseWidgetOffsetY + _BaseWidgetSpanY * index;
    _nameText.setPosition(nameTextPos);

    sf::Vector2f valueRectSize = sf::Vector2f(_BaseValueRectWidth, _BaseValueRectHeight) * uiScale;
    _valueRect.setSize(valueRectSize);
    _valueRect.setOrigin(valueRectSize.x / 2.f, 0.f);
    _valueRect.setFillColor(sf::Color::Transparent);
    _valueRect.setOutlineColor(sf::Color::White);
    float outlineThickness = _BaseValueRectOutlineThickness * uiScale;
    _valueRect.setOutlineThickness(outlineThickness);
    sf::Vector2f valueRectPos;
    valueRectPos.x = float(window.getSize().x) / 2;
    valueRectPos.y = nameTextPos.y + _BaseWidgetOffsetY + nameCharacterSize;
    _valueRect.setPosition(valueRectPos);

    _valueText.setFont(_Font);
    auto valueCharacterSize = (unsigned int)(valueRectSize.y * _ValueTextSizeFactor);
    _valueText.setCharacterSize(valueCharacterSize);
    _valueText.setFillColor(sf::Color::White);

    float triRadius = _BaseTriRadius * uiScale;
    _leftTri = sf::CircleShape(triRadius, 3);
    _leftTri.setOutlineColor(sf::Color::White);
    _leftTri.setOutlineThickness(outlineThickness);
    _leftTri.setFillColor(sf::Color::Transparent);
    _rightTri = sf::CircleShape(triRadius, 3);
    _rightTri.setOutlineColor(sf::Color::White);
    _rightTri.setOutlineThickness(outlineThickness);
    _rightTri.setFillColor(sf::Color::Transparent);
    _leftTri.setOrigin(_leftTri.getRadius(), _leftTri.getRadius());
    _leftTri.setRotation(270);
    _rightTri.setOrigin(_rightTri.getRadius(), _rightTri.getRadius());
    _rightTri.setRotation(90);
    sf::Vector2f leftTriPos, rightTriPos;
    leftTriPos.x = valueRectPos.x - static_cast<int>(valueRectSize.x / 2.f * 1.25);
    rightTriPos.x = valueRectPos.x + static_cast<int>(valueRectSize.x / 2.f * 1.25);
    leftTriPos.y = rightTriPos.y = valueRectPos.y + valueRectSize.y / 2;
    _leftTri.setPosition(leftTriPos);
    _rightTri.setPosition(rightTriPos);

    this->_config = config;
}

void wg::ConfigUIWidget::respond(int x, int y) {
    _respondTri(_leftTri, Left, x, y);
    _respondTri(_rightTri, Right, x, y);
}

void wg::ConfigUIWidget::display(ConfigWindow &window) {
    _valueText.setString(_config->getValue());

    sf::Vector2f valueTextPos;
    sf::Vector2f valueRectPos = _valueRect.getPosition();
    _valueText.setOrigin(_valueText.getLocalBounds().width / 2.f, 0.f);
    valueTextPos.x = valueRectPos.x;
    valueTextPos.y = valueRectPos.y + _valueRect.getSize().y * (0.8f - _ValueTextSizeFactor) / 2.f;
    _valueText.setPosition(valueTextPos);

    window.draw(_nameText);
    window.draw(_leftTri);
    window.draw(_rightTri);
    window.draw(_valueRect);
    window.draw(_valueText);
}

void wg::ConfigUIWidget::_respondTri(const sf::CircleShape &triShape, wg::ConfigUIWidget::TriTag tag, int x, int y) {
    auto pos = triShape.getPosition();
    auto r = triShape.getRadius();
    pos -= {r, r};
    Point a = Point(triShape.getPoint(0) + pos), b = Point(triShape.getPoint(1) + pos), c = Point(triShape.getPoint(2) + pos);
    Triangle tri = Triangle(a, b, c);
    if (tri.contains(Point(x, y))) {
        switch (tag) {
            case Left:
                _config->dec();
                break;
            case Right:
                _config->inc();
                break;
        }
    }
}
