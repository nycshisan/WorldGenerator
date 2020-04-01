//
// Created by nycsh on 2020/2/23.
//

#include "curveInfo.h"

#include "../conf/conf.h"
#include "../misc/misc.h"
#include "data.h"

namespace wg {

//    void CurveInfo::setEndPoints(const Point &pa, const Point &pb) {
//        controlPoints[0] = pa; controlPoints[ControlPointsNum + 1] = pb;
//        auto dp = pa - pb;
//        _vx = dp.y; _vy = -dp.x;
//        float cs = CONF.getBlockEdgesCurveScale();
//        _vx *= cs;
//        _vy *= cs;
//        _minCurveSpanScale = CONF.getBlockEdgesMinCurveScale();
//    }
//
//    Point CurveInfo::randomControlPoints(float begin, float end) {
//        auto h = Random::RandFloat(begin, end);
//
//        auto hp = controlPoints[0] * (1.f - h) + controlPoints[ControlPointsNum + 1] * h;
//        float v = Random::RandFloat(_minCurveSpanScale, 1.f);
//        int sign = Random::RandInt(0, 1);
//        if (sign == 0) v = -v;
//        hp.x += _vx * v;
//        hp.y += _vy * v;
//        auto p = Point(hp);
//        p._resetUIPosition();
//        return p;
//    }
//
//    sf::Vector2f CurveInfo::getCurvePointForDraw(float t) {
//    }

    void CurveInfo::generateSegments(const std::shared_ptr<EdgeInfo> &edge) {
        segments.clear();
        const auto &segmentNumberRange = CONF.getBlockEdgesSegmentNumberRange();
        const auto &spanRange = CONF.getBlockEdgesCurveSpanRange();
        const auto &segmentDistRange = CONF.getBlockEdgesSegmentDistRange();
        const auto &controlPointSpanRange = CONF.getBlockEdgesControlPointSpanRange();
        int sn = Random::RandInt(segmentNumberRange.first, segmentNumberRange.second);

        const Point &pa = (*edge->vertexes.begin())->point, &pb = (*edge->vertexes.rbegin())->point;
        const Point &bca = edge->relatedBlocks.begin()->lock()->center, &bcb = edge->relatedBlocks.rbegin()->lock()->center;
        const auto &pd = pb - pa;
        float vx = pd.y, vy = -pd.x;
        Point v(vx, vy);
        Triangle ta(pa, pb, bca), tb(pa, pb, bcb);
        if (v.dot(bca) < 0) {
            std::swap(ta, tb); // make sure the the first triangle is corresponding to positive v
        }


        std::vector<float> t(sn - 1);
        for (int i = 0; i < sn - 1; ++i) {
            t[i] = Random::RandFloat(0.2f, 0.8f);
        }
        sort(t.begin(), t.end());

        Point endPoints[sn + 1];
        endPoints[0] = pa; endPoints[sn] = pb;
        Triangle *tris[sn + 1]; tris[0] = tris[sn] = nullptr;
        for (int i = 1; i < sn; ++i) {
            auto range = spanRange;
            bool positive = Random::RandBinary();
            while (true) {
                float span = Random::RandFloat(range.first, range.second);
                if (positive) {
                    endPoints[i] = pa + pd * t[i - 1] + v * span;
                    tris[i] = &ta;
                } else {
                    endPoints[i] = pa + pd * t[i - 1] - v * span;
                    tris[i] = &tb;
                }
                if (tris[i]->contains(endPoints[i])) {
                    break;
                } else {
                    range.first /= 2;
                    range.second /= 2;
                }
            }
        }

        segments.resize(sn);
        for (int i = 0; i < sn; ++i) {
            segments[i].controlPoints[0] = endPoints[i];
            segments[i].controlPoints[CurveSegment::ControlPointNumber - 1] = endPoints[i + 1];
        }
        for (int i = 0; i < sn - 1; ++i) {
            auto range = controlPointSpanRange;
            while (true) {
                float span = Random::RandFloat(range.first, range.second);
                const auto &cp = endPoints[i + 1] - pd * span;
                if (tris[i + 1]->contains(cp)) {
                    segments[i].controlPoints[CurveSegment::ControlPointNumber - 2] = cp;
                    break;
                } else {
                    range.first /= 2;
                    range.second /= 2;
                }
            }
            range = controlPointSpanRange;
            while (true) {
                float span = Random::RandFloat(range.first, range.second);
                const auto &cp = endPoints[i + 1] + pd * span;
                if (tris[i + 1]->contains(cp)) {
                    segments[i + 1].controlPoints[1] = cp;
                    break;
                } else {
                    range.first /= 2;
                    range.second /= 2;
                }
            }
            segments[0].controlPoints[1] = segments[0].controlPoints[CurveSegment::ControlPointNumber - 2];
            segments.back().controlPoints[CurveSegment::ControlPointNumber - 2] = segments.back().controlPoints[1];
        }

    }

    Point CurveSegment::getCurvePoint(float t) const {
        const auto &p11 = Point::Lerp(controlPoints[0], controlPoints[1], t),
                   &p12 = Point::Lerp(controlPoints[1], controlPoints[2], t),
                   &p13 = Point::Lerp(controlPoints[2], controlPoints[3], t);
        const auto &p21 = Point::Lerp(p11, p12, t),
                   &p22 = Point::Lerp(p12, p13, t);
        return Point::Lerp(p21, p22, t);
    }
}