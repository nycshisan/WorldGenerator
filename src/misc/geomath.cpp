//
// Created by Nycshisan on 2018/3/17.
//

#include "geomath.h"

#include <cmath>

#include "error.h"

namespace wg {

    Point::Point(float x, float y) : sf::Vector2f(x, y) {
        this->vertex = sf::Vertex(*this);
    }

    Point::Point(const sf::Vector2f &v) : sf::Vector2f(v) {
        this->vertex = sf::Vertex(v);
    }

    float Point::distance(const Point &anoPoint) const {
        float xDist = x - anoPoint.x, yDist = y - anoPoint.y;
        return xDist * xDist + yDist * yDist;
    }

    bool Point::operator==(const Point &anoP) {
        return std::abs(x - anoP.x) < _Error && std::abs(y - anoP.y) < _Error;
    }

    bool Point::operator!=(const Point &anoP) {
        return !((*this) == anoP);
    }

    std::ostream &operator<<(std::ostream &os, const Point &p) {
        os << p.x << " " << p.y;
        return os;
    }

    Triangle::Triangle(const Point &pa, const Point &pb, const Point &pc) {
        points[0] = pa;
        points[1] = pb;
        points[2] = pc;
    }

    bool Triangle::contains(const Point &p) {
        Point &pa = points[0], &pb = points[1], &pc = points[2];
        mat2 pMat = {{{(pb.x - pa.x), (pb.y - pa.y)}, {(pc.x - pa.x), (pc.y - pa.y)}}};
        invert(pMat);
        vec2 pVec = {(p.x - pa.x), (p.y - pa.y)};
        vec2 cVec = pMat * pVec;
        data_t cB = cVec.x, cC = cVec.y;

        return cB > _ContainsError && cC > _ContainsError && cB + cC < 1.0f - _ContainsError;
    }

    Point Triangle::getExCenter() {
        Point &pa = points[0], &pb = points[1], &pc = points[2];
        Line lab = Segment(pa, pb).midPerpendicular(), lac = Segment(pa, pc).midPerpendicular();
        auto result = lab.intersect(lac);
        assertWithSave(!isnan(result.x) && !isnan(result.y));
        return result;
    }

    Line::Line(const Point &pa, const Point &pb) {
        float dx = pa.x - pb.x, dy = pa.y - pb.y;
        if (std::abs(dx) < _err) {
            vertical = true;
            verticalX = pa.x;
            dy = std::signbit(dy) * _err;
        } else if (std::abs(dy) < _err) {
            horizontal = true;
            horizontalY = pa.y;
            dx = std::signbit(dx) * _err;
        }
        _k = dy / dx;
        _b = pa.y - pa.x * _k;
    }

    Line::Line(const Point &p, float k) {
        _k = k;
        _b = p.y - k * p.x;
    }

    Point Line::intersect(const Line &anoLine) {
        float px, py;
        if (horizontal) {
            px = anoLine.xGivenY(horizontalY);
            py = horizontalY;
        } else if (anoLine.horizontal) {
            px = xGivenY(anoLine.horizontalY);
            py = anoLine.horizontalY;
        } else if (vertical) {
            px = verticalX;
            py = anoLine.yGivenX(verticalX);
        } else if (anoLine.vertical) {
            px = anoLine.verticalX;
            py = yGivenX(anoLine.verticalX);
        } else {
            px = (anoLine._b - _b) / (_k - anoLine._k);
            py = yGivenX((px));
        }
        return {px, py};
    }

    float Line::yGivenX(float x) const {
        if (horizontal)
            return horizontalY;
        return _k * x + _b;
    }

    float Line::xGivenY(float y) const {
        if (vertical)
            return verticalX;
        return (y - _b) / _k;
    }

    Line Line::Horizontal(float y) {
        Line r;
        r.horizontal = true;
        r.horizontalY = y;
        return r;
    }

    Line Line::Vertical(float x) {
        Line r;
        r.vertical = true;
        r.verticalX = x;
        return r;
    }

    Segment::Segment(const Point &pa, const Point &pb) : Line(pa, pb) {
        _pa = pa;
        _pb = pb;
    }

    Line Segment::midPerpendicular() {
        Point pMid;
        pMid.x = (_pa.x + _pb.x) / 2.0f;
        pMid.y = (_pa.y + _pb.y) / 2.0f;
        Line r;
        if (vertical) {
            r = Horizontal(pMid.y);
        } else if (horizontal) {
            r = Vertical(pMid.x);
        } else {
            float vk = -1.0f / _k;
            r = Line(pMid, vk);
        }
        return r;
    }

    Rectangle::Rectangle(float left, float right, float top, float down) {
        _left = left;
        _right = right;
        _top = top;
        _down = down;
        _edges[0] = Line(Point(_left, _top), Point(_right, _top));
        _edges[1] = Line(Point(_right, _top), Point(_right, _down));
        _edges[2] = Line(Point(_right, _down), Point(_left, _down));
        _edges[3] = Line(Point(_left, _down), Point(_left, _top));
    }

    bool Rectangle::contains(const Point &p) {
        return p.x >= _left && p.x <= _right && p.y >= _down && p.y <= _top;
    }

    Point Rectangle::intersectRay(const Point &pa, const Point &pb) {
        Line lab(pa, pb);
        sf::Vector2f vab = pa - pb;
        for (auto edge: _edges) {
            Point intersection = lab.intersect(edge);
            if (contains(intersection)) {
                sf::Vector2f vai = pa - intersection;
                float dot = vab.x * vai.x + vab.y * vai.y;
                if (dot >= -_Error) {
                    return intersection;
                }
            }
        }
        assertWithSave(false);
        return {};
    }

    int Rectangle::intersectSegment(const Point &pa, const Point &pb, Point *intersections) {
        int n = 0;
        Line lab(pa, pb);
        for (auto edge: _edges) {
            Point intersection = lab.intersect(edge);
            if (contains(intersection)) {
                sf::Vector2f vai = pa - intersection;
                sf::Vector2f vbi = pb - intersection;
                float dot = vai.x * vbi.x + vai.y * vbi.y;
                if (dot <= _Error) {
                    intersections[n++] = intersection;
                }
            }
        }
        assertWithSave(n == 0 || n == 2);
        return n;
    }

    bool Rectangle::onEdge(const Point &p, Line &line) const {
        for (auto &edge: _edges) {
            if ((edge.vertical && std::abs(edge.verticalX - p.x) < _Error) ||
                (edge.horizontal && std::abs(edge.horizontalY - p.y) < _Error)) {
                line = edge;
                return true;
            }
        }
        return false;
    }

}