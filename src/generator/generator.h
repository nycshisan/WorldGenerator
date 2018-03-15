//
// Created by Nycshisan on 2018/3/8.
//

#ifndef WORLDGENERATOR_GENERATOR_H
#define WORLDGENERATOR_GENERATOR_H

#include "../graphics/window.h"

enum GeneratorState {
    Ready,
    PointsPined
};

class Generator {
    friend void NextButtonResponder(Window &window);

    GeneratorState _state = Ready;
public:
    static Generator &SharedInstance();
};

void NextButtonResponder(Window &window);
void RedoButtonResponder(Window &window);
void SaveButtonResponder(Window &window);

#endif //WORLDGENERATOR_GENERATOR_H
