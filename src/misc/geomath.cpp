//
// Created by Nycshisan on 2018/3/17.
//

#include "geomath.h"

#include <cmath>

Point::Point(float x, float y) : sf::Vector2f(x, y) {}

float Point::distance(const Point &anoPoint) const {
    float xDist = x - anoPoint.x, yDist = y - anoPoint.y;
    return xDist * xDist + yDist * yDist;
}

Triangle::Triangle(const Point &pa, const Point &pb, const Point &pc) {
    points[0] = pa; points[1] = pb; points[2] = pc;
}

bool Triangle::contains(const Point &p) {
    Point &pa = points[0], &pb = points[1], &pc = points[2];
    mat3 pMat = {{{pa.x, pa.y, 1.0f}, {pb.x, pb.y, 1.0f}, {pc.x, pc.y, 1.0f}}};
    invert(pMat);
    vec3 posVec = {p.x, p.y, 1.0f};
    vec3 pVec = pMat * posVec;
    data_t pA = pVec.x, pB = pVec.y, pC = pVec.z;

    return pA > _containsError && pB > _containsError && pC > _containsError;
}

Point Triangle::getExCenter() {
    Point &pa = points[0], &pb = points[1], &pc = points[2];
    Line lab = Segment(pa, pb).midPerpendicular(), lac = Segment(pa, pc).midPerpendicular();
    return lab.intersect(lac);
}

Point Line::intersect(const Line &anoLine) {
    Point p;
    if (_horizontal) {
        p.x = anoLine.xGivenY(_horizontalY);
        p.y = _horizontalY;
    } else if (anoLine._horizontal) {
        p.x = xGivenY(anoLine._horizontalY);
        p.y = anoLine._horizontalY;
    } else if (_vertical) {
        p.x = _verticalX;
        p.y = anoLine.yGivenX(_verticalX);
    } else if (anoLine._vertical) {
        p.x = anoLine._verticalX;
        p.y = yGivenX(anoLine._verticalX);
    } else {
        p.x = (anoLine._b - _b) / (_k - anoLine._k);
        p.y = yGivenX((p.x));
    }
    return p;
}

Line::Line(const Point &pa, const Point &pb) {
    float dx = pa.x - pb.x, dy = pa.y - pb.y;
    if (std::abs(dx) < _err) {
        _vertical = true;
        _verticalX = pa.y;
        dy = std::signbit(dy) * _err;
    }
    else if (std::abs(dy) < _err) {
        _horizontal = true;
        _horizontalY = pa.x;
        dx = std::signbit(dx) * _err;
    }
    _k = dy / dx;
    _b = pa.y - pa.x * _k;
}

Line::Line(const Point &p, float k) {
    _k = k;
    _b = p.y - k * p.x;
}

float Line::yGivenX(float x) const {
    if (_horizontal)
        return _horizontalY;
    return _k * x + _b;
}

float Line::xGivenY(float y) const {
    if (_vertical)
        return _verticalX;
    return (y - _b) / _k;
}

Line Line::Horizontal(float y) {
    Line r;
    r._horizontal = true;
    r._horizontalY = y;
    return r;
}

Line Line::Vertical(float x) {
    Line r;
    r._vertical = true;
    r._verticalX = x;
    return r;
}

Line Segment::midPerpendicular() {
    Point pMid;
    pMid.x = (_pa.x + _pb.x) / 2.0f;
    pMid.y = (_pa.y + _pb.y) / 2.0f;
    Line r;
    if (_vertical) {
        r = Horizontal(pMid.y);
    } else if (_horizontal) {
        r = Vertical(pMid.x);
    } else {
        float vk = -1.0f / _k;
        r = Line(pMid, vk);
    }
    return r;
}
