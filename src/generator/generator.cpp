//
// Created by Nycshisan on 2018/3/8.
//

#include "generator.h"

#include "impl/centers.h"
#include "impl/delaunay.h"
#include "impl/voronoi.h"
#include "impl/lloyd.h"
#include "impl/blocks.h"
#include "impl/blockEdges.h"
#include "impl/distField.h"
#include "impl/heights.h"
#include "impl/mountains.h"
#include "impl/finish.h"

namespace wg {

    Generator::Generator() {
        this->impls.emplace_back(std::make_shared<Centers>());
        this->impls.emplace_back(std::make_shared<DelaunayTriangles>());
        this->impls.emplace_back(std::make_shared<VoronoiDiagram>());
        this->impls.emplace_back(std::make_shared<LloydRelaxation>());
        this->impls.emplace_back(std::make_shared<Blocks>());
        this->impls.emplace_back(std::make_shared<BlockEdges>());
        this->impls.emplace_back(std::make_shared<DistField>());
        this->impls.emplace_back(std::make_shared<Heights>());
        this->impls.emplace_back(std::make_shared<Mountains>());
        this->impls.emplace_back(std::make_shared<Finish>());

        this->_drawer = std::make_shared<Drawer>();
    }

    void Generator::NextButtonResponder(MainWindow &window) {
        // check exit
        if (Generator::SharedInstance().state == Generator::State::Finish) {
            window.close();
        }
        Generator::SharedInstance().next();
        Generator::SharedInstance()._setLabel(window);
    }

    void Generator::RedoButtonResponder(MainWindow &window) {
        Generator::SharedInstance().redo();
    }

    void Generator::UndoButtonResponder(MainWindow &window) {
        Generator::SharedInstance().undo();
        Generator::SharedInstance()._setLabel(window);
    }

    void Generator::SaveButtonResponder(MainWindow &window) {
        const auto &saveResultString = Generator::SharedInstance().save();
        window.setHintLabel(saveResultString);
    }

    void Generator::LoadButtonResponder(MainWindow &window) {
        const auto &loadResultString = Generator::SharedInstance().load();
        window.setHintLabel(loadResultString);
    }

    void Generator::ConfigButtonResponder(MainWindow &window) {
        Generator &generator = Generator::SharedInstance();
        if (generator.state == State::Ready) return;
        if (generator.impls[generator.state]->hasConfigs)
            window.openConfigWindow(&generator);
        else
            window.setHintLabel("No visualizable configuration for this stage.");
    }


    Generator &Generator::SharedInstance() {
        static Generator instance;
        return instance;
    }

    void Generator::_setLabel(MainWindow &window) {
        if (state >= Generator::SharedInstance().impls.size()) return;
        if (state >= 0)
            window.setHintLabel(impls[state]->getHintLabelText());
        else
            window.setHintLabel("Ready!");
    }

    void Generator::display(MainWindow &window) {
        _drawer->setWindow(&window);
        _drawer->commit();
    }

    void Generator::saveErrorData() {
        LOG("Error data saved!");
        impls[State::Centers]->save();
    }

    void Generator::_prepareConfigs() {
        configs.clear();
        impls[state]->getConfigs(*this);
    }

    void Generator::next() {
        if (state >= (int)impls.size() - 1) {
            return; // do nothing if all the generators have been run
        }
        void *input = state >= 0 ? impls[state]->output() : nullptr; // get input from the last generator
        if (CONF.getOutputAutoSave()) {
            if (state >= 0) impls[state]->save(); // save the data of the last generator (is exists)
            else {
                // clear the output directory at the initial stage
                ClearDirectory(CONF.getOutputDirectory());
            }
        }
        state++;
        impls[state]->input(input);
        redo();
    }

    void Generator::redo() {
        _drawer->clear();
        if (state > State::Ready) {
            CONF.reload();
            impls[state]->generate();
            impls[state]->prepareVertexes(*_drawer);
            _prepareConfigs();
        }
    }

    void Generator::undo() {
        _drawer->clear();
        if (state > State::Ready) {
            state--;
            redo();
        }
    }

    std::string Generator::save() {
        if (state > State::Ready)
            return impls[state]->save();
        return "Can't save.";
    }

    std::string Generator::load() {
        if (state > State::Ready) {
            _drawer->clear();
            const auto &loadResult = impls[state]->load();
            impls[state]->prepareVertexes(*_drawer);
            return loadResult;
        }
        return "Can't load.";
    }

}