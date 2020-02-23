//
// Created by nycsh on 2020/2/23.
//

#ifndef WORLDGENERATOR_CURVEINFO_H
#define WORLDGENERATOR_CURVEINFO_H

#include "geomath.h"

namespace wg {

    struct CurveInfo {
        void setEndPoints(const Point& pa, const Point& pb);
        void randomControlPoints();
        sf::Vector2f getCurvePointForDraw(float t);

    private:
        Point _ca, _cb; // Two Bezier control points
        Point _pa, _pb; // Two endpoints
        float _vx{}, _vy{};

        void _setRandomControlPoint(Point& p, float h);
    };

}

#endif //WORLDGENERATOR_CURVEINFO_H
