//
// Created by Nycshisan on 2018/7/4.
//

#ifndef WORLDGENERATOR_GFXMATH_H
#define WORLDGENERATOR_GFXMATH_H

#include <SFML/Graphics.hpp>

sf::Color operator * (const sf::Color &color, float x);
void operator *= (sf::Color &color, float x);

sf::Color ColorBlend(const sf::Color &colorA, const sf::Color &colorB, float factor);

#endif //WORLDGENERATOR_GFXMATH_H
