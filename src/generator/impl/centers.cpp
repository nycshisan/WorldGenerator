//
// Created by Nycshisan on 2018/3/18.
//

#include "centers.h"

#include <random>
#include <fstream>

namespace wg {

    void Centers::input(void* inputData) { assert(inputData == nullptr); }

    void Centers::generate() {
        int width = CONF.getMapWidth(), height = CONF.getMapHeight();
        int span = CONF.getCenterSpan();
        int randomSeed = CONF.getMapRandomSeed();

        int n = CONF.getCenterNumber();
        int padding = CONF.getCenterPadding();

        std::mt19937 gen(randomSeed);
        std::uniform_int_distribution<> disX(padding, width - padding), disY(padding, height - padding);

        _centers.clear();
        for (int i = 0; i < n;) {
            int x = disX(gen);
            int y = disY(gen);
            bool occupiedFlag = true;
            for (auto &point : _centers) {
                if (abs(x - int(point.x)) <= span && abs(y - int(point.y)) <= span) {
                    occupiedFlag = false;
                    break;
                }
            }
            if (occupiedFlag) {
                _centers.emplace_back(Point(x, y));
                ++i;
            }
        }
    }

    void* Centers::output() {
        return (void*)&_centers;
    }

    void Centers::prepareVertexes(Drawer &drawer) {
        for (auto &point : _centers) {
            drawer.appendVertex(sf::Points, point.vertex);
        }
    }

    std::string Centers::save() {
        const auto &fp = CONF.getOutputDirectory() + CONF.getCentersOutputPath();
        CreateDependentDirectory(fp);
        std::ofstream outfile(fp);
        if (!outfile.good()) return "Centers saving failed.";
        for (auto &center: _centers) {
            outfile << (int) center.x << " " << (int) center.y << std::endl;
        }
        return "Centers saved.";
    }

    std::string Centers::load() {
        const auto &fp = CONF.getOutputDirectory() + CONF.getCentersOutputPath();
        std::ifstream infile(fp);
        if (!infile.good()) return "Centers loading failed.";
        std::vector<Point> centers;
        while (!infile.eof()) {
            int x = 0, y = 0;
            infile >> x >> y;
            centers.emplace_back(Point(x, y));
        }
        centers.pop_back();
        _centers = centers;
        return "Centers loaded.";
    }

    std::string Centers::getHintLabelText() {
        return "Generated block centers.";
    }

}