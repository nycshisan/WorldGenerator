//
// Created by Nycshisan on 2018/3/8.
//

#ifndef WORLDGENERATOR_GENERATOR_H
#define WORLDGENERATOR_GENERATOR_H

#include "centers.h"
#include "delaunay.h"
#include "voronoi.h"
#include "lloyd.h"
#include "blocks.h"
#include "coast.h"

#include "config.h"

namespace wg {

    class MainWindow;
    class Drawer;

    enum GeneratorState {
        Ready,
        Centers,
        DelaunayTriangles,
        VoronoiDiagram,
        LloydRelaxation,
        Blocks,
        Coast,
        HeatDist
    };

    class Generator {

        Generator();

        void _nextState();

        void _lastState();

        void _setLabel(MainWindow &window);

        std::shared_ptr<Drawer> _drawer;

        class Centers _blockCenters;
        class DelaunayTriangles _delaunayTriangles;
        class VoronoiDiagram _voronoiDiagram;
        class LloydRelaxation _lloydRelaxation;
        class Blocks _blocks;
        class Coast _coast;

        void _prepareConfigs();

    public:
        GeneratorState state = Ready;

        std::vector<std::shared_ptr<GeneratorConfig>> configs;

        static Generator &SharedInstance();

        void display(MainWindow &window);

        void redo();

        void saveErrorData();

        static void NextButtonResponder(MainWindow &window);

        static void RedoButtonResponder(MainWindow &window);

        static void UndoButtonResponder(MainWindow &window);

        static void SaveButtonResponder(MainWindow &window);

        static void LoadButtonResponder(MainWindow &window);

        static void ConfigButtonResponder(MainWindow &window);
    };

}

#endif //WORLDGENERATOR_GENERATOR_H
