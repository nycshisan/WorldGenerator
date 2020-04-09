//
// Created by nycsh on 2020/2/23.
//

#include "curveInfo.h"

#include "../conf/conf.h"
#include "../misc/misc.h"
#include "data.h"

namespace wg {

    void CurveInfo::generateSegments(const std::shared_ptr<EdgeInfo> &edge) {
        segments.clear();
        endPoints.clear();
        const auto &segmentNumberRange = CONF.getBlockEdgesSegmentNumberRange();
        const auto &spanRange = CONF.getBlockEdgesCurveSpanRange();
        auto segmentDistRange = CONF.getBlockEdgesSegmentDistRange();
        const auto &controlPointSpanRange = CONF.getBlockEdgesControlPointSpanRange();
        const float minSegmentInterval = CONF.getBlockEdgesMinSegmentInterval();
        int sn = Random::RandInt(segmentNumberRange.first, segmentNumberRange.second);

        const Point &pa = (*edge->vertexes.begin())->point, &pb = (*edge->vertexes.rbegin())->point;
        const Point &pm = (pa + pb) / 2.f;
        const Point &bca = edge->relatedBlocks.begin()->lock()->center, &bcb = edge->relatedBlocks.rbegin()->lock()->center;
        const auto &pd = pb - pa;
        float vx = pd.y, vy = -pd.x;
        Point v(vx, vy);
        Triangle ta(pa, pb, bca), tb(pa, pb, bcb);
        if (v.dot(bca - pm) < 0) {
            std::swap(ta, tb); // make sure the the first triangle is corresponding to positive v
        }

        // generate the t points
        std::vector<float> t(sn - 1);
        segmentDistRange.second -= float(sn - 1) * minSegmentInterval;
        for (int i = 0; i < sn - 1; ++i) {
            t[i] = Random::RandFloat(segmentDistRange);
        }
        sort(t.begin(), t.end());
        for (int i = 0; i < t.size(); ++i) {
            t[i] += float(i) * minSegmentInterval;
        }

        // generate endpoints
        endPoints.resize(sn + 1);
        endPoints[0] = pa; endPoints[sn] = pb;
        Triangle *tris[sn + 1]; tris[0] = tris[sn] = nullptr;
        for (int i = 1; i < sn; ++i) {
            auto range = spanRange;
            bool positive = Random::RandBinary();
            while (true) {
                float span = Random::RandFloat(range);
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

        // generate segments
        segments.resize(sn);
        for (int i = 0; i < sn; ++i) {
            segments[i].controlPoints[0] = endPoints[i];
            segments[i].controlPoints[CurveSegment::ControlPointNumber - 1] = endPoints[i + 1];
        }
        for (int i = 0; i < sn - 1; ++i) {
            auto range = controlPointSpanRange;
            while (true) {
                float span = Random::RandFloat(range);
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
                float span = Random::RandFloat(range);
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