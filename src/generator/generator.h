//
// Created by Nycshisan on 2018/3/8.
//

#ifndef WORLDGENERATOR_GENERATOR_H
#define WORLDGENERATOR_GENERATOR_H

#include "../graphics/window.h"
#include "centers.h"
#include "delaunay.h"
#include "voronoi.h"

enum GeneratorState {
    Ready,
    BlockCenters,
    DelaunayTriangles,
    VoronoiDiagram,
};

class Generator {
    friend void NextButtonResponder(Window &window);
    friend void RedoButtonResponder(Window &window);
    friend void UndoButtonResponder(Window &window);
    friend void SaveButtonResponder(Window &window);

    GeneratorState _state = Ready;

    void _nextState();

    class BlockCenters _blockCenters;
    BlockCenters::Output _centers;
    class DelaunayTriangles _delaunayTriangles;

public:

    static Generator &SharedInstance();

    void display(Window &window);
};

void NextButtonResponder(Window &window);
void RedoButtonResponder(Window &window);
void UndoButtonResponder(Window &window);
void SaveButtonResponder(Window &window);

#endif //WORLDGENERATOR_GENERATOR_H
