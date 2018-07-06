//
// Created by Nycshisan on 2018/3/18.
//

#include "centers.h"

#include <random>
#include <fstream>

#include "../conf/conf.h"
#include "../graphics/drawer.h"

namespace wg {

    void Centers::input() {}

    void Centers::generate() {
        int width = CONF.getMapWidth(), height = CONF.getMapHeight();
        int span = CONF.getCenterSpan();
        int randomSeed = CONF.getMapRandomSeed();

        unsigned char occupied[width][height];
        std::memset(occupied, 0, static_cast<size_t>(width * height));

        int n = CONF.getCenterNumber();
        int padding = CONF.getCenterPadding();

        std::mt19937 gen(randomSeed);
        std::uniform_int_distribution<> disX(padding, width - padding), disY(padding, height - padding);

        _centers.clear();
        for (int i = 0; i < n;) {
            int x = disX(gen);
            int y = disY(gen);
            bool occupiedFlag = false;
            for (int j = std::max(x - span, 0); j <= std::min(x + span, width - 1); ++j) {
                for (int k = std::max(y - span, 0); k <= std::min(y + span, height - 1); ++k) {
                    if (occupied[j][k]) {
                        occupiedFlag = true;
                    }
                }
            }
            if (occupiedFlag) {
                continue;
            }
            _centers.emplace_back(Point(x, y));
            for (int j = std::max(x - span, 0); j <= std::min(x + span, width - 1); ++j) {
                for (int k = std::max(y - span, 0); k <= std::min(y + span, height - 1); ++k) {
                    occupied[j][k] = 1;
                }
            }
            ++i;
        }
    }

    Centers::Output Centers::output() {
        return _centers;
    }

    void Centers::prepareVertexes(Drawer &drawer) {
        for (auto &point : _centers) {
            drawer.appendVertex(sf::Points, point.vertex);
        }
    }

    void Centers::save() {
        std::ofstream outfile("logs/centers.txt", std::ios_base::trunc);
        for (auto &center: _centers) {
            outfile << (int) center.x << " " << (int) center.y << std::endl;
        }
    }

    void Centers::load() {
        std::ifstream infile("logs/centers.txt");
        std::vector<Point> centers;
        while (!infile.eof()) {
            int x, y;
            infile >> x >> y;
            centers.emplace_back(Point(x, y));
        }
        centers.pop_back();
        _centers = centers;
    }

}