//
// Created by Nycshisan on 2018/3/17.
//

#include "geomath.h"

#include <cmath>

float pointDistance(Point pa, Point pb) {
    float xDist = pa.x - pb.x, yDist = pa.y - pb.y;
    return xDist * xDist + yDist * yDist;
}

data_t triangleContainsError = -0.0001f; // Some input may cause problem if the error is 0. I do not know why...
bool triangleContains(const Point &pa, const Point &pb, const Point &pc, const Point &p) {
    mat3 pMat = {{{pa.x, pa.y, 1.0f},
                  {pb.x, pb.y, 1.0f},
                  {pc.x, pc.y, 1.0f}}};
    invert(pMat);
    vec3 posVec = {p.x, p.y, 1.0f};
    vec3 pVec = pMat * posVec;
    data_t pA = pVec.x, pB = pVec.y, pC = pVec.z;

    return pA > triangleContainsError && pB > triangleContainsError && pC > triangleContainsError;
}

Point triangleExCenter(const Point &pa, const Point &pb, const Point &pc) {
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

Line Segment::midPerpendicular() {
    Point pMid;
    pMid.x = (_pa.x + _pb.x) / 2.0f;
    pMid.y = (_pa.y + _pb.y) / 2.0f;
    Line lab = Line(_pa, _pb), r;
    if (lab._vertical) {
        r._horizontal = true;
        r._horizontalY = pMid.y;
    } else if (lab._horizontal) {
        r._vertical = true;
        r._verticalX = pMid.x;
    } else {
        float vk = -1.0f / lab._k;
        r = Line(pMid, vk);
    }
    return r;
}
