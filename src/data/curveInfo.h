//
// Created by nycsh on 2020/2/23.
//

#ifndef WORLDGENERATOR_CURVEINFO_H
#define WORLDGENERATOR_CURVEINFO_H

#include <memory>

#include "geomath.h"

namespace wg {

    struct EdgeInfo;

    struct CurveSegment {
        static constexpr int ControlPointNumber = 4;

        Point controlPoints[ControlPointNumber];

        Point getCurvePoint(float t) const;
    };

    struct CurveInfo {
        std::vector<CurveSegment> segments;

        void generateSegments(const std::shared_ptr<EdgeInfo> &edge);
    };

}

#endif //WORLDGENERATOR_CURVEINFO_H
