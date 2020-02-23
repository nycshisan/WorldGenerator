//
// Created by nycsh on 2020/2/23.
//

#include "curveInfo.h"

#include "../conf/conf.h"
#include "../misc/misc.h"

namespace wg {

    void CurveInfo::setEndPoints(const Point &pa, const Point &pb) {
        _pa = pa; _pb = pb;
        auto dp = _pa - _pb;
        _vx = dp.y; _vy = -dp.x;
        float vl = std::hypot(_vx, _vy);
        float cs = CONF.getBlockEdgesCurveSpan() * CONF.getUIMapScaleConversion();
        _vx *= (cs / vl);
        _vy *= (cs / vl);
    }

    void CurveInfo::randomControlPoints() {
        auto ha = Random::RandFloat(0, 1);
        auto hb = Random::RandFloat(0, 1);
        if (ha > hb) {
            std::swap(ha, hb);
        }

        _setRandomControlPoint(_ca, ha);
        _setRandomControlPoint(_cb, hb);
    }

    sf::Vector2f CurveInfo::getCurvePointForDraw(float t) {
        const auto &ma = _pa.vertex.position * (1.f - t) + _ca.vertex.position * t;
        const auto &mb = _cb.vertex.position * (1.f - t) + _pb.vertex.position * t;
        return ma * (1.f - t) + mb * t;
    }

    void CurveInfo::_setRandomControlPoint(Point& p, float h) {
        auto hp = _pa * (1.f - h) + _pb * h;
        float v = Random::RandFloat(0.6f, 1.f);
        int sign = Random::RandInt(0, 1);
        if (sign == 0) v = -v;
        hp.x += _vx * v;
        hp.y += _vy * v;
        p = Point(hp);
        p.resetUIPosition();
    }
}