//
// Created by Nycshisan on 2018/4/8.
//

#include "error.h"

#include "../generator/generator.h"

namespace wg {

    void SaveErrorData() {
        Generator::SharedInstance().saveErrorData();
    }

}