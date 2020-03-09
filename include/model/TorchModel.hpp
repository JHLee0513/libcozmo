////////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2019,  Brian Lee, Vinitha Ranganeni
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

#ifndef INCLUDE_MODEL_TORCHMODEL_HPP_
#define INCLUDE_MODEL_TORCHMODEL_HPP_

#include <Eigen/Dense>
#include <torch/torch.h>
#include <string>
#include "model/Model.hpp"
// #include <ATen/ATen.h>

namespace libcozmo {
namespace model {

// Simple Linear Model implemented in Torch
struct LinearRegression : torch::nn::Module {
  LinearRegression() {
    fc = register_module("fc", torch::nn::Linear(3, 2));
  }

  // Implement the Net's algorithm.
  torch::Tensor forward(torch::Tensor x) {
    // Use one of many tensor manipulation functions.
    x = fc->forward(x);
    return x;
  }

  // Use one of many "standard library" modules.
  torch::nn::Linear fc{nullptr};
};

/// LinearRegression model for all model types. When using this class or its derived
/// classes in a script you must wrap the code with Py_Initialize() and
/// Py_Finalize();
class TorchModel : public Model{
 public:
    /// Constructs instance of Torch-based transition model.
    TorchModel();

    /// Gets the end state after applying the given action on the input state
    /// Input vectors vary based on derived class
    ///
    /// \param input_action Given action vector
    /// \param input_state Given state vector
    /// \param[out] output_state vector
    /// \return True if prediction successfully calculated; false otherwise;
    bool predict_state(
        const Eigen::VectorXd& input_action,
        const Eigen::VectorXd& input_state,
        Eigen::VectorXd* output_state) const override;

    /// Given path to weights loads network with model
    bool load_weights(const std::string path);
 
 private:
    const std::shared_ptr<torch::nn::Module> m_model;
};

}  // namespace model
}  // namespace libcozmo

#endif  // INCLUDE_MODEL_MODEL_HPP_
