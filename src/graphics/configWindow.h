//
// Created by Nycshisan on 2018/7/5.
//

#ifndef WORLDGENERATOR_CONFIGWINDOW_H
#define WORLDGENERATOR_CONFIGWINDOW_H

#include <SFML/Graphics.hpp>

#include "../generator/generator.h"

namespace wg {

    class Generator;
    class ConfigWindow;

    class ConfigUIWidget {
        static sf::Font _Font;
        static bool _FontLoaded;
        static constexpr float _BaseTriRadius = 10.f;
        static constexpr float _BaseValueRectWidth = 120.f, _BaseValueRectHeight = 30.f;
        static constexpr float _ValueTextSizeFactor = 0.6f;
        static constexpr float _BaseValueRectOutlineThickness = 2.f;
        static constexpr float _BaseWidgetOffsetY = 20.f, _BaseWidgetSpanY = 140.f;

        sf::CircleShape _leftTri, _rightTri;
        sf::Text _valueText, _nameText;
        sf::RectangleShape _valueRect;

        std::shared_ptr<GeneratorConfig> _config;

        enum TriTag {
            Left, Right
        };

        void _respondTri(const sf::CircleShape &triShape, wg::ConfigUIWidget::TriTag tag, int x, int y);

    public:
        explicit ConfigUIWidget(const std::shared_ptr<GeneratorConfig> &config, int index, const ConfigWindow &window);

        void respond(int x, int y);
        void display(ConfigWindow &window);
    };

    class ConfigWindow : public sf::RenderWindow {
        static constexpr int _BaseConfigWindowWidth = 220;

        Generator *_generator;

        GeneratorState _lastState;

        std::vector<ConfigUIWidget> _configWidgets;


    public:
        explicit ConfigWindow(Generator *generator);

        void play();
    };

}

#endif //WORLDGENERATOR_CONFIGWINDOW_H
