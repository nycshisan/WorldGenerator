//
// Created by Nycshisan on 2018/3/8.
//

#ifndef WORLDGENERATOR_CONF_H
#define WORLDGENERATOR_CONF_H

#include <fstream>
#include <vector>

#include "document.h"
#include "istreamwrapper.h"
#include "ostreamwrapper.h"
#include "prettywriter.h"
#include "pointer.h"

namespace wg {

    class Configure : public rapidjson::Document {
        static const std::string conf_fn;

        Configure();

    public:
        static Configure &SharedInstance();

        void reload();

        template <typename T>
        void save(const std::string &pointerPath, T value) {
            rapidjson::Pointer pointer(pointerPath.c_str());
            rapidjson::Value *vp = pointer.Get(*this);
            assert(vp != nullptr);
            vp->Set(value);

            std::ofstream ofs(conf_fn);
            rapidjson::OStreamWrapper osw(ofs);
            rapidjson::PrettyWriter<rapidjson::OStreamWrapper> writer(osw);

            Accept(writer);
        }

        int getMapWidth() const;

        int getMapHeight() const;

        int getMapRandomSeed() const;

        unsigned getUIMapScaleConversion() const;

        unsigned getUIMapWidth() const;

        unsigned getUIMapHeight() const;

        int getUIUpdateFPSFrameInterval() const;

        float getUIScale() const;

        std::string getUIFontFilename() const;

        int getCenterNumber() const;

        int getCenterPadding() const;

        int getCenterSpan() const;

        std::string getCentersOutputPath() const;

        bool getDelaunayShowBoundingTriangles() const;

        float getLloydFactor() const;

        int getLloydIteration() const;

        std::string getBlocksOutputPath() const;

        int getCoastContinentNumber() const;

        float getCoastOceanFactor() const;

        float getCoastSeaFactor() const;

        float getCoastNoiseInfluence() const;
    };

}

#define CONF wg::Configure::SharedInstance()

#endif //WORLDGENERATOR_CONF_H
