//
// Created by Nycshisan on 2018/3/8.
//

#include "conf.h"

const std::string Configure::conf_fn = "conf/conf.json";

Configure::Configure() {
    std::ifstream ifs(conf_fn);
    rapidjson::IStreamWrapper isw(ifs);
    ParseStream(isw);
}

const Configure &Configure::SharedInstance() {
    static Configure instance;
    return instance;
}

