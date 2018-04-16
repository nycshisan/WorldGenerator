//
// Created by Nycshisan on 2018/3/8.
//

#ifndef WORLDGENERATOR_GENERATOR_H
#define WORLDGENERATOR_GENERATOR_H

#include <memory>

#include "../graphics/window.h"
#include "../graphics/drawer.h"
#include "centers.h"
#include "delaunay.h"
#include "voronoi.h"
#include "lloyd.h"
#include "blocks.h"
#include "coast.h"

enum GeneratorState {
    Ready,
    BlockCenters,
    DelaunayTriangles,
    VoronoiDiagram,
    LloydRelaxation,
    Blocks,
    Coast,
};

class Generator {
    GeneratorState _state = Ready;

    void _nextState();
    void _lastState();
    void _setLabel(Window &window);

    std::shared_ptr<Drawer> _drawer;

    class BlockCenters _blockCenters;
    BlockCenters::Output _centers;
    class DelaunayTriangles _delaunayTriangles;
    DelaunayTriangles::Output _tris;
    class VoronoiDiagram _voronoiDiagram;
    VoronoiDiagram::Output _vd;
    class LloydRelaxation _lloydRelaxation;
    LloydRelaxation::Output _relaxedVd;
    class Blocks _blocks;

public:
    static Generator &SharedInstance();

    void display(Window &window);

    static void NextButtonResponder(Window &window);
    static void RedoButtonResponder(Window &window);
    static void UndoButtonResponder(Window &window);
    static void SaveButtonResponder(Window &window);
    static void LoadButtonResponder(Window &window);

    void SaveErrorData();
};

#endif //WORLDGENERATOR_GENERATOR_H
