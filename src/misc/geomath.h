//
// Created by Nycshisan on 2018/3/17.
//

#ifndef WORLDGENERATOR_GEOMATH_H
#define WORLDGENERATOR_GEOMATH_H

#include "SFML/Graphics.hpp"

#include "simd.h"

typedef sf::Vector2f Point;

float pointDistance(Point pa, Point pb);

bool triangleContains(const Point &pa, const Point &pb, const Point &pc, const Point &p);

class Line {
    friend class Segment;

    float _err = 1e-3;
    float _k = 0.0f, _b = 0.0f;
    bool _vertical = false; float _verticalX = 0.0f;
    bool _horizontal = false; float _horizontalY = 0.0f;

public:
    Line() = default;
    Line(const Point &pa, const Point &pb);
    Line(const Point &p, float k);

    Point intersect(const Line &anoLine);
    float yGivenX(float x) const;
    float xGivenY(float y) const;
};

class Segment : public Line {
    Point _pa, _pb;
public:
    Segment(const Point &pa, const Point &pb) : Line(pa, pb) {
        _pa = pa; _pb = pb;
    }

    Line midPerpendicular();
};

Point triangleExCenter(const Point &pa, const Point &pb, const Point &pc);

#endif //WORLDGENERATOR_GEOMATH_H
