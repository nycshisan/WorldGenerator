//
// Created by Nycshisan on 2018/4/15.
//

#include "coast.h"

#include <cmath>
#include <limits>

#include "../conf/conf.h"

namespace wg {

    void Coast::input(const Input &input) {
        _blockInfos = input;
    }

    void Coast::generate() {
        int randomSeed = CONF.getMapRandomSeed();
        int continentNumber = CONF.getCoastContinentNumber();
        float oceanFactor = CONF.getCoastOceanFactor();
        float seaFactor = CONF.getCoastSeaFactor();
        int width = CONF.getMapWidth(), height = CONF.getMapHeight();
        float minContinentCenterDist = float(width * width + height * height) / (continentNumber * continentNumber);
        float noiseInfluence = CONF.getCoastNoiseInfluence();

        std::mt19937 rg(randomSeed);
        for (auto &blockInfo: _blockInfos) {
            blockInfo->coastType = BlockInfo::CoastType::Land;
            blockInfo->isContinentCenter = false;
        }

        std::set<std::shared_ptr<BlockInfo>, std::owner_less<std::shared_ptr<BlockInfo>>> continentCenters;
        std::uniform_int_distribution<> dis(0, int(_blockInfos.size()) - 1);
        while (continentCenters.size() < continentNumber) {
            auto possibleCenter = _blockInfos[dis(rg)];
            bool isValid = true;
            for (auto &center: continentCenters) {
                if (center->center.distance(possibleCenter->center) < minContinentCenterDist)
                    isValid = false;
            }
            if (isValid) {
                possibleCenter->isContinentCenter = true;
                continentCenters.emplace(possibleCenter);
            }
        }

        for (auto &block: _blockInfos) {
            auto pos = block->center;
            float pn = NoiseGenerator::PerlinNoise(pos.x, pos.y);
            float minDist1 = std::numeric_limits<float>::max(), minDist2 = std::numeric_limits<float>::max();
            float maxDist = std::numeric_limits<float>::min();
            for (auto &center: continentCenters) {
                float dist = pos.distance(center->center);
                if (dist < minDist1) {
                    minDist2 = minDist1;
                    minDist1 = dist;
                } else if (dist < minDist2) {
                    minDist2 = dist;
                }
                if (dist > maxDist) {
                    maxDist = dist;
                }
            }
            minDist1 /= minContinentCenterDist;
            minDist2 /= minContinentCenterDist;
            float noise;
            if (continentNumber > 1)
                noise = pn * (noiseInfluence + minDist1 - std::abs(minDist1 - minDist2)) / noiseInfluence;
            else
                noise = (pn * noiseInfluence + minDist1) / (noiseInfluence + 1.0f);
            if (noise > oceanFactor)
                block->coastType = BlockInfo::CoastType::Ocean;
            else if (noise > seaFactor)
                block->coastType = BlockInfo::CoastType::Sea;
        }

        for (auto &block: _blockInfos) {
            if (block->coastType == BlockInfo::CoastType::Land) {
                for (auto &edge: block->edges) {
                    for (auto &relatedBlock: edge->relatedBlocks) {
                        if (relatedBlock.lock() != block->thisPtr.lock()) {
                            if (relatedBlock.lock()->coastType == BlockInfo::CoastType::Ocean) {
                                block->coastType = BlockInfo::CoastType::Sea;
                            }
                        }
                    }
                }
            };
        }
    }

    void Coast::prepareVertexes(Drawer &drawer) {
        for (auto &blockInfo: _blockInfos) {
            if (blockInfo->coastType == BlockInfo::CoastType::Ocean) {
                _prepareBlockVertexes(drawer, blockInfo, sf::Color::Blue);
            }
            sf::Color seaColor;
            seaColor.a = (sf::Color::Blue.a + sf::Color::White.a) / (unsigned char) (2);
            seaColor.g = (sf::Color::Blue.g + sf::Color::White.g) / (unsigned char) (2);
            seaColor.b = (sf::Color::Blue.b + sf::Color::White.b) / (unsigned char) (2);
            if (blockInfo->coastType == BlockInfo::CoastType::Sea) {
                _prepareBlockVertexes(drawer, blockInfo, seaColor);
            }
            if (blockInfo->isContinentCenter) {
                _prepareBlockVertexes(drawer, blockInfo, sf::Color::White);
            }
            for (auto &edgeInfo: blockInfo->edges) {
                drawer.appendVertex(sf::Lines, (*edgeInfo->vertexes.begin())->point.vertex);
                drawer.appendVertex(sf::Lines, (*edgeInfo->vertexes.rbegin())->point.vertex);
            }
        }
    }

}