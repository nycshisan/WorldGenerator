//
// Created by Nycshisan on 2018/7/6.
//

#include "config.h"

#include <cmath>

#include "../conf/conf.h"
#include "generator.h"

wg::GeneratorConfigFloat::GeneratorConfigFloat(const std::string &name, int min, int max, int value,
                                               float factor, const std::string &pointerPath)
        : wg::GeneratorConfig(name, min, max, value, pointerPath) {
    this->_factor = factor;
}

std::string wg::GeneratorConfigFloat::getValue() {
    auto reservedDigitNumber = static_cast<int>(std::ceilf(std::log10f(1.f / _factor)));
    std::string format = "%." + std::to_string(reservedDigitNumber) + "f";
    char buf[30];
    sprintf(buf, format.c_str(), _value * _factor);
    return std::string(buf);
}

void wg::GeneratorConfigFloat::saveConfig() {
    CONF.save(_pointerPath, _value * _factor);
    Generator::SharedInstance().redo();
}

wg::GeneratorConfig::GeneratorConfig(const std::string &name, int min, int max, int value, const std::string &pointerPath) {
    this->name = name;
    this->_min = min;
    this->_max = max;
    this->_value = value;
    this->_pointerPath = pointerPath;
}

void wg::GeneratorConfig::inc() {
    _value = std::min(_value + 1, _max);
    saveConfig();
}

void wg::GeneratorConfig::dec() {
    _value = std::max(_value - 1, _min);
    saveConfig();
}
