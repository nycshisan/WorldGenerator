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

const Configure &Configure::SharedInstance() {
    static Configure instance;
    return instance;
}

unsigned int Configure::getMapWidth() const {
    return (*this)["map"]["width"].GetUint();
}

unsigned int Configure::getMapHeight() const {
    return (*this)["map"]["height"].GetUint();
}

unsigned int Configure::getUIBarHeight() const {
    return (*this)["ui"]["barHeight"].GetUint();
}

unsigned int Configure::getUIBarSeparatorHeight() const {
    return (*this)["ui"]["barSeparatorHeight"].GetUint();
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

