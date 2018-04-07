//
// Created by Nycshisan on 2018/3/17.
//

#ifndef WORLDGENERATOR_GEOMATH_H
#define WORLDGENERATOR_GEOMATH_H

#include "SFML/Graphics.hpp"

#include "simd.h"

class Point : public sf::Vector2f {
public:
    Point() = default;
    Point(float x, float y);

    float distance(const Point &anoPoint) const;
};

class Triangle {
    data_t _containsError = -0.001f; // Some input may cause problem if the error is 0. I do not know why...

public:
    Point points[3];

    Triangle(const Point &pa, const Point &pb, const Point &pc);

    bool contains(const Point &p);
    Point getExCenter();
};

class Line {
protected:
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

    static Line Horizontal(float y);
    static Line Vertical(float x);
};

class Segment : public Line {
    Point _pa, _pb;

public:
    Segment(const Point &pa, const Point &pb);

    Line midPerpendicular();
};

class Rectangle {
    float _left, _right, _top, _down;
    Line _edges[4];

public:

    Rectangle() = default;
    explicit Rectangle(float left, float right, float top, float down);

    bool contains(const Point &p);
    Point intersects(const Point &pa, const Point &pb);
};

#endif //WORLDGENERATOR_GEOMATH_H
