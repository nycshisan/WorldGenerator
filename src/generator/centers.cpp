//
// Created by Nycshisan on 2018/3/18.
//

#include "centers.h"

#include <random>

#include "../conf/conf.h"

void BlockCenters::input(unsigned int width, unsigned int height) {
    _width = width; _height = height;
    _pointShape.setRadius(CONF.getUIPointRadius());
}

void BlockCenters::generate() {
    int n = CONF.getCenterNumber();
    int padding = CONF.getCenterPadding();

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> disX(padding, _width - padding), disY(padding, _height - padding);

    _centers.clear();
    for (int i = 0; i < n; ++i) {
        int x = disX(gen);
        int y = disY(gen);
        _centers.emplace_back(Point(x, y));
    }
}

BlockCenters::Output BlockCenters::output() {
    return _centers;
}

void BlockCenters::draw(Window &window) {
    for (auto point : _centers) {
        _pointShape.setPosition(point);
        window.draw(_pointShape);
    }
}

void BlockCenters::save() {
    std::ofstream outfile("logs/centers.txt");
    for (auto &center: _centers) {
        outfile << (int)center.x << " " << (int)center.y << std::endl;
    }
}

void BlockCenters::load() {
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
