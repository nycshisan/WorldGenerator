//
// Created by nycsh on 2020/2/14.
//

#include <queue>

#include "blockEdges.h"

#include "blocks.h"

bool DrawStraightEdge = true;
bool DrawControlPoints = true;

namespace wg {

    std::string BlockEdges::getHintLabelText() {
        return "Bezier block edges generated.";
    }

    void BlockEdges::generate() {
        _blockInfos = *(Blocks::Output *) _inputData;

        Random::ResetRandomEngine();

        _edges.clear();

        for (const auto &block: _blockInfos) {
            for (const auto &edge: block->edges) {
                if (!edge->isMargin) {
                    _edges.emplace(edge);
                }
            }
        }

        for (const auto &edge : _edges) {
            edge->curveInfo.generateSegments(edge);
        }

        _outputData = (void*)&_blockInfos;
    }

    void BlockEdges::prepareVertexes(Drawer &drawer) {
        sf::Vertex curvePoint;
        curvePoint.color = sf::Color::Green;
        float step = CONF.getBlockEdgesCurveStep();
        for (const auto &edge : _edges) {
            if (DrawStraightEdge) {
                drawer.appendVertex(sf::Lines, (*edge->vertexes.begin())->point.vertexUI);
                drawer.appendVertex(sf::Lines, (*edge->vertexes.rbegin())->point.vertexUI);
            }
            if (DrawControlPoints) {
                for (const auto &segment : edge->curveInfo.segments) {
                    for (const auto &p : segment.controlPoints) {
                        drawer.appendCustomPointShape(p.vertexUI, sf::Color::Red, 1.5);
                    }
                }
            }
            for (const auto &segment : edge->curveInfo.segments) {
                float t = 0;
                while (t < 1) {
                    curvePoint.position = segment.getCurvePoint(t).vertexUI.position;
                    drawer.appendVertex(sf::Points, curvePoint);
                    t += step;
                }
            }
        }
    }
}
