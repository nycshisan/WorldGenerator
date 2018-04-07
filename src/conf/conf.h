//
// Created by Nycshisan on 2018/3/8.
//

#ifndef WORLDGENERATOR_CONF_H
#define WORLDGENERATOR_CONF_H

#include <fstream>

#include "document.h"
#include "istreamwrapper.h"

class Configure : public rapidjson::Document {
    static const std::string conf_fn;

    Configure();

public:
    static const Configure& SharedInstance();

    int getMapWidth() const;
    int getMapHeight() const;
    int getUIBarHeight() const;
    int getUIBarSeparatorHeight() const;
    std::string getUIFontFilename() const;
    float getUIPointRadius() const;
    int getCenterNumber() const;
    int getCenterPadding() const;
    int getCenterSpan() const;
    float getLloydFactor() const;
    int getLloydIteration() const;
};

#define CONF Configure::SharedInstance()

#endif //WORLDGENERATOR_CONF_H
