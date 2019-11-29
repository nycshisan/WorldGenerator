//
// Created by Nycshisan on 2018/7/4.
//

#include "gfxmath.h"

sf::Color operator * (const sf::Color &color, float x) {
    sf::Color rc;
    rc.r = sf::Uint8 (color.r * x);
    rc.g = sf::Uint8 (color.g * x);
    rc.b = sf::Uint8 (color.b * x);
    rc.a = color.a;
    return rc;
}

void operator *= (sf::Color &color, float x) {
    color = color * x;
}

sf::Color ColorBlend(const sf::Color &colorA, const sf::Color &colorB, float factor) {
    return colorA * factor + colorB * (1 - factor);
}
