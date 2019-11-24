//
// Created by Nycshisan on 2018/3/8.
//

#include "generator.h"

#include "../misc/log.h"
#include "../conf/conf.h"
#include "../graphics/window.h"
#include "../graphics/drawer.h"

namespace wg {

    Generator::Generator() {
        this->_drawer = std::make_shared<Drawer>();
    }

    void Generator::NextButtonResponder(MainWindow &window) {
        Generator &generator = Generator::SharedInstance();
        if (generator.state != Coast)
            generator._drawer->clearVertexes();
        switch (generator.state) {
            case Ready:
                generator._blockCenters.input();
                generator._blockCenters.generate();
                generator._blockCenters.prepareVertexes(*generator._drawer);
                break;
            case Centers:
                generator._delaunayTriangles.input(generator._blockCenters.output());
                generator._delaunayTriangles.generate();
                generator._delaunayTriangles.prepareVertexes(*generator._drawer);
                break;
            case DelaunayTriangles:
                generator._voronoiDiagram.input(generator._blockCenters.output(), generator._delaunayTriangles.output());
                generator._voronoiDiagram.generate();
                generator._voronoiDiagram.prepareVertexes(*generator._drawer);
                break;
            case VoronoiDiagram:
                generator._lloydRelaxation.input(generator._voronoiDiagram.output());
                generator._lloydRelaxation.generate();
                generator._lloydRelaxation.prepareVertexes(*generator._drawer);
                break;
            case LloydRelaxation:
                generator._blocks.input(generator._lloydRelaxation.output());
                generator._blocks.generate();
                generator._blocks.prepareVertexes(*generator._drawer);
                break;
            case Blocks:
                generator._coast.input(generator._blocks.output());
                generator._coast.generate();
                generator._coast.prepareVertexes(*generator._drawer);
                break;
            default:
                break;
        }
        generator._nextState();
        generator._setLabel(window);
    }

    void Generator::RedoButtonResponder(MainWindow &window) {
        CONF.reload();
        Generator &generator = Generator::SharedInstance();
        generator.redo();
    }

    void Generator::UndoButtonResponder(MainWindow &window) {
        Generator &generator = Generator::SharedInstance();
        generator._lastState();
        RedoButtonResponder(window);
        generator._setLabel(window);
    }

    void Generator::SaveButtonResponder(MainWindow &window) {
        Generator &generator = Generator::SharedInstance();
        switch (generator.state) {
            case Centers:
                if (generator._blockCenters.save())
                    window.setHintLabel("Centers saved.");
                else
                    window.setHintLabel("Centers saving failed.");
                break;
            default:
                window.setHintLabel("Can't save.");
                generator.saveErrorData();
                break;
        }
    }

    void Generator::LoadButtonResponder(MainWindow &window) {
        Generator &generator = Generator::SharedInstance();
        switch (generator.state) {
            case Centers:
                if (generator._blockCenters.load())
                    window.setHintLabel("Centers loaded.");
                else
                    window.setHintLabel("Centers loading failed.");
                break;
            default:
                window.setHintLabel("Can't load.");
                break;
        }
    }

    void Generator::ConfigButtonResponder(MainWindow &window) {
        Generator &generator = Generator::SharedInstance();
        switch (generator.state) {
            case Coast:
                window.openConfigWindow(&generator);
                break;
            default:
                window.setHintLabel("No visualizable configuration for this stage.");
                break;
        }
    }


    Generator &Generator::SharedInstance() {
        static Generator instance;
        return instance;
    }

    void Generator::_nextState() {
        switch (state) {
            case Ready:
                state = Centers;
                break;
            case Centers:
                state = DelaunayTriangles;
                break;
            case DelaunayTriangles:
                state = VoronoiDiagram;
                break;
            case VoronoiDiagram:
                state = LloydRelaxation;
                break;
            case LloydRelaxation:
                state = Blocks;
                break;
            case Blocks:
                state = Coast;
                break;
            default:
                break;
        }
        _prepareConfigs();
    }

    void Generator::_lastState() {
        switch (state) {
            case Centers:
                state = Ready;
                break;
            case DelaunayTriangles:
                state = Centers;
                break;
            case VoronoiDiagram:
                state = DelaunayTriangles;
                break;
            case LloydRelaxation:
                state = VoronoiDiagram;
                break;
            case Blocks:
                state = LloydRelaxation;
                break;
            case Coast:
                state = Blocks;
                break;
            default:
                break;
        }
        _prepareConfigs();
    }

    void Generator::_setLabel(MainWindow &window) {
        switch (state) {
            case Ready:
                window.setHintLabel("Ready!");
                break;
            case Centers:
                window.setHintLabel("Generated block centers.");
                break;
            case DelaunayTriangles:
                window.setHintLabel("Generated Delaunay triangles.");
                break;
            case VoronoiDiagram:
                window.setHintLabel("Generated Voronoi diagram.");
                break;
            case LloydRelaxation:
                window.setHintLabel("Done Lloyd relaxation.");
                break;
            case Blocks:
                window.setHintLabel("Initialized blocks' information.");
                break;
            case Coast:
                window.setHintLabel("Generated the coast.");
                break;
            default:
                break;
        }
    }

    void Generator::display(MainWindow &window) {
        _drawer->setWindow(&window);
        _drawer->commit();
    }

    void Generator::saveErrorData() {
        _blockCenters.save();
    }

    void Generator::_prepareConfigs() {
        configs.clear();
        switch (state) {
            case Coast:
                _coast.getConfigs(configs);
                break;
            default:
                break;
        }
    }

    void Generator::redo() {
        _drawer->clearVertexes();
        switch (state) {
            case Ready:
                break;
            case Centers:
                _blockCenters.generate();
                _blockCenters.prepareVertexes(*_drawer);
                break;
            case DelaunayTriangles:
                _delaunayTriangles.generate();
                _delaunayTriangles.prepareVertexes(*_drawer);
                break;
            case VoronoiDiagram:
                _voronoiDiagram.generate();
                _voronoiDiagram.prepareVertexes(*_drawer);
                break;
            case LloydRelaxation:
                _lloydRelaxation.generate();
                _lloydRelaxation.prepareVertexes(*_drawer);
                break;
            case Blocks:
                _blocks.generate();
                _blocks.prepareVertexes(*_drawer);
                break;
            case Coast:
                _coast.generate();
                _coast.prepareVertexes(*_drawer);
                break;
            default:
                LOGERR("Invalid generator state!");
        }
    }

}