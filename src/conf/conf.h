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

#define NO_DISCARD [[nodiscard]]

        NO_DISCARD int getMapWidth() const;

        NO_DISCARD int getMapHeight() const;

        NO_DISCARD int getMapRandomSeed() const;

        NO_DISCARD unsigned getUIMapScaleConversion() const;

        NO_DISCARD unsigned getUIMapWidth() const;

        NO_DISCARD unsigned getUIMapHeight() const;

        NO_DISCARD int getUIUpdateFPSFrameInterval() const;

        NO_DISCARD float getUIScale() const;

        NO_DISCARD std::string getUIFontFilename() const;

        NO_DISCARD std::string getOutputDirectory() const;

        NO_DISCARD bool getOutputAutoSave() const;

        NO_DISCARD bool getInstallEnable() const;

        NO_DISCARD std::string getInstallTarget() const;

        NO_DISCARD int getCenterNumber() const;

        NO_DISCARD int getCenterPadding() const;

        NO_DISCARD int getCenterSpan() const;

        NO_DISCARD std::string getCentersOutputPath() const;

        NO_DISCARD bool getDelaunayShowBoundingTriangles() const;

        NO_DISCARD float getLloydFactor() const;

        NO_DISCARD int getLloydIteration() const;

        NO_DISCARD std::string getBlocksOutputPath() const;

        NO_DISCARD unsigned getHeightMapWidth() const;

        NO_DISCARD unsigned getHeightMapHeight() const;

        NO_DISCARD std::string getHeightMapOutputPath() const;
    };

}

#define CONF wg::Configure::SharedInstance()

#endif //WORLDGENERATOR_CONF_H
