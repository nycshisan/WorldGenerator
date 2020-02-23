//
// Created by nycsh on 2020/2/14.
//

#include "blockEdges.h"

#include "blocks.h"

std::string wg::BlockEdges::getHintLabelText() {
    return "Bezier block edges generated.";
}

void wg::BlockEdges::generate() {
    _blockInfos = *(Blocks::Output*)_inputData;

    Random::ResetRandomEngine();

    _edges.clear();

    for (const auto &block: _blockInfos) {
        for (const auto &edge: block->edges) {
            _edges.emplace(edge);
        }
    }

    for (const auto& edge: _edges) {
        if (!edge->isMargin) {
            edge->curveInfo.setEndPoints((*edge->vertexes.begin())->point, (*edge->vertexes.rbegin())->point);
            edge->curveInfo.randomControlPoints();
        }
    }

    _outputData = (void*)&_blockInfos;
}

void wg::BlockEdges::prepareVertexes(wg::Drawer &drawer) {
    sf::Vertex curvePoint;
    curvePoint.color = sf::Color::Green;
    float step = CONF.getBlockEdgesCurveStep();
    for (const auto &edge: _edges) {
//        drawer.appendVertex(sf::Lines, (*edge->vertexes.begin())->point.vertex);
//        drawer.appendVertex(sf::Lines, (*edge->vertexes.rbegin())->point.vertex);
        if (!edge->isMargin) {
            float t = 0;
            while (t < 1) {
                curvePoint.position = edge->curveInfo.getCurvePointForDraw(t);
                drawer.appendVertex(sf::Points, curvePoint);
                t += step;
            }
        }
    }
}
