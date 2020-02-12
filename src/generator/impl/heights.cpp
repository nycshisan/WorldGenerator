//
// Created by nycsh on 2020/2/10.
//

#include "heights.h"

#include "blocks.h"

namespace wg {

    void Heights::generate() {
        auto &blockInfos = *(Blocks::Output*)_inputData;

        auto heightMapWidth = CONF.getHeightMapWidth(), heightMapHeight = CONF.getHeightMapHeight();

        _blockHeightInfos.first = blockInfos;
        _blockHeightInfos.second.resize(heightMapHeight, std::vector<float>(heightMapWidth, 0.5f));

        BlockHeightInfoDrawable::createTexture(CONF.getUIMapWidth(), CONF.getUIMapHeight());
        BlockHeightInfoDrawable::setTexture(_blockHeightInfos.second);

        _outputData = (void*)&_blockHeightInfos;
    }

    void Heights::prepareVertexes(Drawer &drawer) {
        BlockHeightInfoDrawable::drawHeightsTo(drawer, _blockHeightInfos.first);
    }

    std::string Heights::getHintLabelText() {
        return "Initialized the height map.";
    }

    void BlockHeightInfoDrawable::createTexture(unsigned int width, unsigned int height) {
        _t.create(width, height);
        _s.setTexture(_t);
        float scaleX = float(CONF.getUIMapWidth()) / float(width), scaleY = float(CONF.getUIMapHeight()) / float(width);
        _s.setScale(scaleX, scaleY);
    }

    void BlockHeightInfoDrawable::setTexture(const std::vector<std::vector<float>> &heights) {
        auto h = heights.size(), w = heights[0].size();
        auto size = h * w;
        auto rgba = new sf::Uint8[size * 4]; int bi = 0;
        for (int i = 0; i < h; ++i) {
            for (int j = 0; j < w; ++j) {
                auto rgb = sf::Uint8(256 * heights[i][j]);
                rgba[bi++] = rgb;
                rgba[bi++] = rgb;
                rgba[bi++] = rgb;
                rgba[bi++] = 255; // alpha channel
            }
        }
        _t.update(rgba);
        delete[] rgba;
    }

    void BlockHeightInfoDrawable::drawHeightsTo(Drawer &drawer, const std::vector<std::shared_ptr<BlockInfo>> &blockInfos) {
        for (const auto &block : blockInfos) {
            for (const auto &edge : block->edges) {
                for (const auto &point : edge->vertexes) {
                    auto v = point->point.vertex;
                    v.color = sf::Color::Blue;
                    drawer.appendVertex(sf::Lines, v);
                }
            }
        }

        drawer.addSprite(_s);
    }
}