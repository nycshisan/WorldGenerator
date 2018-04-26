//
// Created by Nycshisan on 2018/4/15.
//

#include "coast.h"

#include "../conf/conf.h"

void Coast::input(Coast::Input input) {
    _blockInfos = std::move(input);
    _k = int(CONF.getCenterNumber() * CONF.getCoastOceanProportion());
    _randomSeed = CONF.getMapRandomSeed();
}

void Coast::generate() {
    for (auto &blockInfo: _blockInfos) {
        blockInfo->isOcean = false;
    }
    std::vector<int> blockIndices;
    for (int i = 0; i < _blockInfos.size(); ++i) {
        blockIndices.emplace_back(i);
    }
    _gen = std::mt19937(_randomSeed);
    _findOceanBlocks(blockIndices, 0, int(blockIndices.size()), _k);
}

void Coast::draw(Drawer &drawer) {
    for (auto &blockInfo: _blockInfos) {
        if (blockInfo->isOcean) {
            drawer.draw(*blockInfo, sf::Color::Blue);
        }
        for (auto &edgeInfo: blockInfo->edges) {
            drawer.draw((*edgeInfo));
        }
    }
}

void Coast::_findOceanBlocks(std::vector<int> &indices, int begin, int size, int k) {
    if (size == 0)
        return;
    std::uniform_int_distribution<> dis(begin, begin + size - 1);
    std::swap(indices[begin], indices[dis(_gen)]);
    float pivot = _blockInfos[indices[begin]]->area;
    int pivotPos = 0; // Also means there are how many blocks larger than the pivot
    for (int i = 1; i < size; ++i) {
        if (_blockInfos[indices[begin + i]]->area > pivot) {
            ++pivotPos;
            std::swap(indices[begin + pivotPos], indices[begin + i]);
        }
    }
    std::swap(indices[begin], indices[begin + pivotPos]);
    if (k < pivotPos) {
        _findOceanBlocks(indices, begin, pivotPos, k);
    } else {
        for (int i = 0; i < pivotPos; ++i) {
            _blockInfos[indices[i]]->isOcean = true;
        }
        if (k == pivotPos + 1) {
            _blockInfos[indices[pivotPos]]->isOcean = true;
        } else {
            _findOceanBlocks(indices, begin + pivotPos + 1, size - pivotPos - 1, k - pivotPos - 1);
        }
    }
}
