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

    int Configure::getCenterNumber() const {
        return (*this)["centers"]["number"].GetInt();
    }

    int Configure::getCenterPadding() const {
        return (*this)["centers"]["padding"].GetInt();
    }

    int Configure::getCenterSpan() const {
        return (*this)["centers"]["span"].GetInt();
    }

    std::string Configure::getCentersOutputPath() const {
        return (*this)["centers"]["outputPath"].GetString();
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

    std::string Configure::getBlocksOutputPath() const {
        return (*this)["blocks"]["outputPath"].GetString();
    }

    int Configure::getCoastContinentNumber() const {
        return (*this)["coast"]["continentNumber"].GetInt();
    }

    float Configure::getCoastOceanFactor() const {
        return (*this)["coast"]["oceanFactor"].GetFloat();
    }

    float Configure::getCoastSeaFactor() const {
        return (*this)["coast"]["seaFactor"].GetFloat();
    }

    float Configure::getCoastNoiseInfluence() const {
        return (*this)["coast"]["noiseInfluence"].GetFloat();
    }

    unsigned Configure::getUIMapWidth() const {
        return getMapWidth() / getUIMapScaleConversion();
    }

    unsigned Configure::getUIMapHeight() const {
        return getMapHeight() / getUIMapScaleConversion();
    }

}
