//
// Created by Nycshisan on 2018/3/8.
//

#include "conf.h"

namespace wg {

    const std::string Configure::conf_fn = "conf/conf.json";

    Configure::Configure() {
        std::ifstream ifs(conf_fn);
        rapidjson::IStreamWrapper isw(ifs);
        ParseStream(isw);
    }

    Configure &Configure::SharedInstance() {
        static Configure instance;
        return instance;
    }

    void Configure::reload() {
        Configure::SharedInstance() = Configure();
    }

    std::string Configure::getModuleOutputPath(const std::string &moduleName) {
        return (*this)[moduleName.c_str()]["outputPath"].GetString();
    }

    int Configure::getMapWidth() const {
        return (*this)["map"]["width"].GetInt();
    }

    int Configure::getMapHeight() const {
        return (*this)["map"]["height"].GetInt();
    }

    int Configure::getMapRandomSeed() const {
        return (*this)["map"]["randomSeed"].GetInt();
    }

    unsigned Configure::getUIMapScaleConversion() const {
        return (*this)["ui"]["mapScaleInversion"].GetUint();
    }

    int Configure::getUIUpdateFPSFrameInterval() const {
        return (*this)["ui"]["updateFPSFrameInterval"].GetInt();
    }

    float Configure::getUIScale() const {
        return (*this)["ui"]["scale"].GetFloat();
    }

    std::string Configure::getUIFontFilename() const {
        return (*this)["ui"]["font"].GetString();
    }

    unsigned Configure::getUIMapWidth() const {
        return getMapWidth() / getUIMapScaleConversion();
    }

    unsigned Configure::getUIMapHeight() const {
        return getMapHeight() / getUIMapScaleConversion();
    }

    std::string Configure::getOutputDirectory() const {
        return (*this)["output"]["directory"].GetString();
    }

    bool Configure::getOutputAutoSave() const {
        return (*this)["output"]["autosave"].GetBool();
    }

    bool Configure::getInstallEnable() const {
        return (*this)["install"]["enable"].GetBool();
    }

    std::string Configure::getInstallTarget() const {
        return (*this)["install"]["target"].GetString();
    }

    int Configure::getCenterNumber() const {
        return (*this)["centers"]["number"].GetInt();
    }

    int Configure::getCenterPadding() const {
        return (*this)["centers"]["padding"].GetInt();
    }

    int Configure::getCenterSpan() const {
        return (*this)["centers"]["span"].GetInt();
    }

    bool Configure::getDelaunayShowBoundingTriangles() const {
        return (*this)["delaunay"]["showBoundingTriangles"].GetBool();
    }

    float Configure::getLloydFactor() const {
        return (*this)["lloyd"]["factor"].GetFloat();
    }

    int Configure::getLloydIteration() const {
        return (*this)["lloyd"]["iteration"].GetInt();
    }

    float Configure::getBlockEdgesCurveSpan() const {
        return (*this)["blockEdges"]["curveSpan"].GetFloat();
    }

    float Configure::getBlockEdgesCurveStep() const {
        return (*this)["blockEdges"]["curveStep"].GetFloat();
    }

    int Configure::getDistFieldSize() const {
        return (*this)["distField"]["size"].GetInt();
    }

    unsigned Configure::getHeightMapWidth() const {
        return (*this)["heightMap"]["width"].GetUint();
    }

    unsigned Configure::getHeightMapHeight() const {
        return (*this)["heightMap"]["height"].GetUint();
    }

}
