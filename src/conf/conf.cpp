//
// Created by Nycshisan on 2018/3/8.
//

#include "conf.h"

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

int Configure::getUIBarHeight() const {
    return (*this)["ui"]["barHeight"].GetInt();
}

int Configure::getUIBarSeparatorHeight() const {
    return (*this)["ui"]["barSeparatorHeight"].GetInt();
}

std::string Configure::getUIFontFilename() const {
    return (*this)["ui"]["font"].GetString();
}

float Configure::getUIPointRadius() const {
    return (*this)["ui"]["pointRadius"].GetFloat();
}

int Configure::getCenterNumber() const {
    return (*this)["centers"]["number"].GetInt();
}

int Configure::getCenterPadding() const {
    return (*this)["centers"]["padding"].GetInt();
}

int Configure::getCenterSpan() const {
    return (*this)["centers"]["span"].GetUint();
}

float Configure::getLloydFactor() const {
    return (*this)["lloyd"]["factor"].GetFloat();
}

int Configure::getLloydIteration() const {
    return (*this)["lloyd"]["iteration"].GetInt();
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