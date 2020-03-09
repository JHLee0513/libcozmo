////////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2019,  Vinitha Ranganeni
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
//     1. Redistributions of source code must retain the above copyright notice
//        this list of conditions and the following disclaimer.
//     2. Redistributions in binary form must reproduce the above copyright
//        notice, this list of conditions and the following disclaimer in the
//        documentation and/or other materials provided with the distribution.
//     3. Neither the name of the copyright holder nor the names of its
//        contributors may be used to endorse or promote products derived from
//        this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
////////////////////////////////////////////////////////////////////////////////

#include "model/TorchModel.hpp"
#include <sstream>
#include <iostream>
#include "utils/utils.hpp"

namespace libcozmo {
namespace model {

TorchModel::TorchModel() : \
    m_model(std::make_shared<LinearRegression>()) {}

bool TorchModel::predict_state(
        const Eigen::VectorXd& input_action,
        const Eigen::VectorXd& input_state,
        Eigen::VectorXd* output_state) const {
    if (input_action.size() != 4 || input_state.size() != 3) {
        return false;
    }

    // auto X = torch::tensor({input_action[0], input_action[0], input_action[0], input_action[3]}, 
    //     torch::requires_grad(false).dtype(torch::kFloat32))
    //     .view({4,1});

    // auto Y = m_model->forward(X);

    // double distance = static_cast<double>(Y.data[0]);
    // double dtheta = static_cast<double>(Y.data[0]);

    // double x = input_state[0] + distance * cos(dtheta);
    // double y = input_state[1] + distance * sin(dtheta);
    // double theta = input_state[2] + dtheta;
    // *output_state = Eigen::Vector3d(x, y, theta);
    return true;
}

bool TorchModel::load_weights(const std::string path) {
    // try {
    //     torch::load(m_model, path);
    //     return true;
    // } catch (const char* msg) {
    //     cerr << msg << std::cout::endl;
    //     return false;
    // }
}

}  // namespace model
}  // namespace libcozmo
