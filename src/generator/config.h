//
// Created by Nycshisan on 2018/7/6.
//

#ifndef WORLDGENERATOR_CONFIG_H
#define WORLDGENERATOR_CONFIG_H

#include <string>
#include <vector>

namespace wg {
    class GeneratorConfig {
    protected:
        float _min, _max, _value;
        std::string _pointerPath;

    public:
        GeneratorConfig(const std::string &name, int min, int max, int value, const std::string &pointerPath);

        std::string name;

        virtual std::string getValue() = 0;
        virtual void saveConfig() = 0;

        void inc();
        void dec();
    };

    class GeneratorConfigFloat : public GeneratorConfig {
        float _factor;
    public:
        GeneratorConfigFloat(const std::string &name, int min, int max, int value, float factor, const std::string &pointerPath);

        std::string getValue() override;
        void saveConfig() override;
    };
}

#endif //WORLDGENERATOR_CONFIG_H
