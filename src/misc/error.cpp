//
// Created by Nycshisan on 2018/4/8.
//

#include "error.h"

#include "../generator/generator.h"

void SaveErrorData() {
    Generator::SharedInstance().SaveErrorData();
}
