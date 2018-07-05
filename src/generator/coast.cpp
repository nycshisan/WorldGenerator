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

        int rs = CONF.getCoastUseStaticRandomSeed() ? randomSeed : std::random_device()();

        std::mt19937 rg(rs);
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

        std::vector<float> minDists;
        float maxMinDist = std::numeric_limits<float>::min();
        for (auto &block: _blockInfos) {
            auto pos = block->center;
            float minDist = std::numeric_limits<float>::max();
            for (auto &center: continentCenters) {
                float dist = pos.distance(center->center);
                if (dist < minDist) {
                    minDist = dist;
                }
            }
            if (minDist > maxMinDist) {
                maxMinDist = minDist;
            }
            minDists.emplace_back(minDist);
        }
        for (int i = 0; i < minDists.size(); ++i) {
            auto &block = _blockInfos[i];
            auto pos = block->center;
            float pn = NoiseGenerator::PerlinNoise(pos.x, pos.y);
            float noise;
            float minDistFactor = minDists[i] / maxMinDist;
            noise = pn * noiseInfluence + minDistFactor * (1.0f - noiseInfluence);
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
            sf::Color oceanColor = sf::Color::Blue;
            if (blockInfo->coastType == BlockInfo::CoastType::Ocean) {
                _prepareBlockVertexes(drawer, blockInfo, oceanColor);
            }

            sf::Color seaColor = ColorBlend(sf::Color::White, sf::Color::Blue, 0.5);
            if (blockInfo->coastType == BlockInfo::CoastType::Sea) {
                _prepareBlockVertexes(drawer, blockInfo, seaColor);
            }

            sf::Color landColor = sf::Color(0x66, 0x33, 0x00);
            if (blockInfo->coastType == BlockInfo::CoastType::Land) {
                _prepareBlockVertexes(drawer, blockInfo, landColor);
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