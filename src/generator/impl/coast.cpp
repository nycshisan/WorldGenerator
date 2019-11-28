//
// Created by Nycshisan on 2018/4/15.
//

#include "coast.h"

#include <cmath>
#include <limits>

#include "../../conf/conf.h"
#include "../../graphics/drawer.h"
#include "../generator.h"

namespace wg {

    Coast::Coast() {
        this->hasConfigs = true;
    }

    void Coast::input(void* inputData) {
        _blockInfos = *(Input*)inputData;
    }

    void Coast::generate() {
        int randomSeed = CONF.getMapRandomSeed();
        int continentNumber = CONF.getCoastContinentNumber();
        float oceanFactor = CONF.getCoastOceanFactor();
        float seaFactor = CONF.getCoastSeaFactor();
        int width = CONF.getMapWidth(), height = CONF.getMapHeight();
        float minContinentCenterDist = sqrtf(width * width + height * height) / continentNumber;
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
                if (center->center.squareDistance(possibleCenter->center) < minContinentCenterDist)
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
                float dist = pos.squareDistance(center->center);
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
            float minDistFactor = minDists[i] / maxMinDist;
            float noise = pn * noiseInfluence + minDistFactor * (1.0f - noiseInfluence);
            if (noise > oceanFactor)
                block->coastType = BlockInfo::CoastType::Ocean;
            else if (noise > seaFactor)
                block->coastType = BlockInfo::CoastType::Sea;
        }

        for (auto &block: _blockInfos) {
            if (block->coastType == BlockInfo::CoastType::Land) {
                for (auto &edge: block->edges) {
                    for (auto &relatedBlock: edge->relatedBlocks) {
                        if (relatedBlock.lock() != block) {
                            if (relatedBlock.lock()->coastType == BlockInfo::CoastType::Ocean) {
                                block->coastType = BlockInfo::CoastType::Sea;
                            }
                        }
                    }
                }
            };
        }
    }

    void *Coast::output() {
        return nullptr;
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

        for (auto &blockInfo: _blockInfos) {
            if (blockInfo->coastType == BlockInfo::CoastType::Land) {
                for (auto &edgeInfo: blockInfo->edges) {
                    auto &relatedBlocks = edgeInfo->relatedBlocks;
                    if (relatedBlocks.size() > 1) {
                        for (auto &relatedBlock: relatedBlocks) {
                            if (relatedBlock.lock() != blockInfo &&
                                relatedBlock.lock()->coastType != BlockInfo::CoastType::Land) {
                                _prepareCoast(drawer, edgeInfo);
                            }
                        }
                    }
                }
            }
        }
    }

    void Coast::getConfigs(Generator &generator) {
        auto &configs = generator.configs;
        std::shared_ptr<GeneratorConfig> config;
        auto ofi = static_cast<int>(CONF.getCoastOceanFactor() * 100);
        auto sfi = static_cast<int>(CONF.getCoastSeaFactor() * 100);
        auto nii = static_cast<int>(CONF.getCoastNoiseInfluence() * 100);
        config = std::make_shared<GeneratorConfigFloat>("Ocean Factor", 0, 100, ofi, 0.01 + 1e-4, "/coast/oceanFactor");
        configs.emplace_back(config);
        config = std::make_shared<GeneratorConfigFloat>("Sea Factor", 0, 100, sfi, 0.01 + 1e-4, "/coast/seaFactor");
        configs.emplace_back(config);
        config = std::make_shared<GeneratorConfigFloat>("Noise Influence", 0, 100, nii, 0.01 + 1e-4, "/coast/noiseInfluence");
        configs.emplace_back(config);
    }

    std::string Coast::getHintLabelText() {
        return "Generated the coast.";
    }

}