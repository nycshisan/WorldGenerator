//
// Created by Nycshisan on 2018/3/18.
//

#include "centers.h"

namespace wg {

    void Centers::generate() {
        assert(_inputData == nullptr);

        Random::ResetRandomEngine();

        int width = CONF.getMapWidth(), height = CONF.getMapHeight();
        int span = CONF.getCenterSpan();

        int n = CONF.getCenterNumber();
        int padding = CONF.getCenterPadding();

        _centers.clear();
        for (int i = 0; i < n;) {
            int x = Random::RandInt(padding, width - padding);
            int y = Random::RandInt(padding, height - padding);
            bool occupiedFlag = true;
            for (auto &point : _centers) {
                if (abs(x - int(point.x)) <= span && abs(y - int(point.y)) <= span) {
                    occupiedFlag = false;
                    break;
                }
            }
            if (occupiedFlag) {
                _centers.emplace_back(Point(float(x), float(y)));
                ++i;
            }
        }

        _outputData = (void*)&_centers;
    }

    void Centers::prepareVertexes(Drawer &drawer) {
        for (auto &point : _centers) {
            drawer.appendPointShape(point.vertex);
        }
    }

    std::string Centers::save() {
        const auto &fp = CONF.getOutputDirectory() + CONF.getModuleOutputPath("centers");
        CreateDependentDirectory(fp);
        std::ofstream outfile(fp);
        if (!outfile.good()) return "Centers saving failed.";
        for (auto &center: _centers) {
            outfile << (int) center.x << " " << (int) center.y << std::endl;
        }
        return "Centers saved.";
    }

    std::string Centers::load() {
        const auto &fp = CONF.getOutputDirectory() + CONF.getModuleOutputPath("centers");
        std::ifstream infile(fp);
        if (!infile.good()) return "Centers loading failed.";
        std::vector<Point> centers;
        while (!infile.eof()) {
            int x = 0, y = 0;
            infile >> x >> y;
            centers.emplace_back(Point(float(x), float(y)));
        }
        centers.pop_back();
        _centers = centers;
        return "Centers loaded.";
    }

    std::string Centers::getHintLabelText() {
        return "Generated block centers.";
    }

}