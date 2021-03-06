////////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2019, Vinitha Ranganeni, Brian Lee
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

#include <gtest/gtest.h>
#include "statespace/SE2.hpp"

namespace libcozmo {
namespace statespace {
namespace test {

class SE2StatespaceTest: public ::testing::Test {
 public:
    SE2StatespaceTest() : \
        statespace(0.1, 8), resolution(0.1), num_theta_vals(8) {}

    void SetUp() {
        statespace.get_or_create_state(SE2::State(3, 2, 1));
        statespace.get_or_create_state(SE2::State(1, 3, 3));
    }

    void TearDown() {}

    ~SE2StatespaceTest() {}

    SE2 statespace;
    aikido::statespace::SE2 continuous_statespace;
    const double resolution;
    const int num_theta_vals;
};

TEST_F(SE2StatespaceTest, GetsOrCreatesDiscreteState) {
    // Cheking existing states IDs are returned correctly
    EXPECT_EQ(0, statespace.get_or_create_state(SE2::State(3, 2, 1)));
    EXPECT_EQ(1, statespace.get_or_create_state(SE2::State(1, 3, 3)));

    // Checking that new state is created and appropiate ID is returned
    EXPECT_EQ(2, statespace.get_or_create_state(SE2::State(1, 1, 1)));
}

TEST_F(SE2StatespaceTest, GetsOrCreatesContinuousState) {
    // Checking that new state is created and appropiate ID is returned
    aikido::statespace::SE2::State state;
    continuous_statespace.expMap(Eigen::Vector3d(1, 1, 1), &state);
    EXPECT_EQ(2, statespace.get_or_create_state(state));
}

TEST_F(SE2StatespaceTest, GetsOrCreatesVectorState) {
    // Checking that new state is created and appropiate ID is returned
    Eigen::VectorXd state_vector(3);
    state_vector << 1, 1, 1;
    EXPECT_EQ(2, statespace.get_or_create_state(state_vector));
}

TEST_F(SE2StatespaceTest, GetsOrCreatesContinuousStateException) {
    // Checking for exception when creating state via vector
    Eigen::VectorXd state_vector(4);
    state_vector << 1, 1, 1, 1;
    EXPECT_THROW(
        statespace.get_or_create_state(state_vector),
        std::runtime_error);
}

TEST_F(SE2StatespaceTest, DiscreteToContinuousStateConversion) {
    aikido::statespace::SE2::State out_state;
    statespace.discrete_state_to_continuous(SE2::State(1, 2, 1), &out_state);

    Eigen::VectorXd log_state;
    continuous_statespace.logMap(&out_state, log_state);
    EXPECT_DOUBLE_EQ(log_state[0], 0.15);
    EXPECT_DOUBLE_EQ(log_state[1], 0.25);
    EXPECT_DOUBLE_EQ(log_state[2], M_PI/4);
}

TEST_F(SE2StatespaceTest, ContinuousToDiscreteStateConversion) {
    aikido::statespace::SE2::State in_state;
    continuous_statespace.expMap(
        Eigen::Vector3d(0.17, 0.257, M_PI/5), &in_state);

    SE2::State out_state;
    statespace.continuous_state_to_discrete(in_state, &out_state);

    EXPECT_DOUBLE_EQ(out_state.X(), 1);
    EXPECT_DOUBLE_EQ(out_state.Y(), 2);
    EXPECT_DOUBLE_EQ(out_state.Theta(), 1);
}

TEST_F(SE2StatespaceTest, GetsStateID) {
    int state_id;
    EXPECT_TRUE(statespace.get_state_id(SE2::State(3, 2, 1), &state_id));
    EXPECT_EQ(state_id, 0);

    EXPECT_TRUE(statespace.get_state_id(SE2::State(1, 3, 3), &state_id));
    EXPECT_EQ(state_id, 1);

    EXPECT_FALSE(statespace.get_state_id(SE2::State(1, 0, 3), &state_id));
}

TEST_F(SE2StatespaceTest, GetsState) {
    const SE2::State* s1 = static_cast<SE2::State*>(statespace.get_state(0));
    EXPECT_EQ(3, s1->X());
    EXPECT_EQ(2, s1->Y());
    EXPECT_EQ(1, s1->Theta());

    const SE2::State* s2 = static_cast<SE2::State*>(statespace.get_state(1));
    EXPECT_EQ(1, s2->X());
    EXPECT_EQ(3, s2->Y());
    EXPECT_EQ(3, s2->Theta());

    EXPECT_EQ(nullptr, statespace.get_state(2));
}

TEST_F(SE2StatespaceTest, IsValidState) {
    EXPECT_TRUE(statespace.is_valid_state(SE2::State(1, 1, 0)));
    EXPECT_TRUE(statespace.is_valid_state(SE2::State(1, 1, 7)));
    EXPECT_FALSE(statespace.is_valid_state(SE2::State(1, 1, 8)));
}

TEST_F(SE2StatespaceTest, StateSpaceSize) {
    EXPECT_EQ(statespace.size(), 2);
}

TEST_F(SE2StatespaceTest, GetsDistanceBetweenDiscreteStates) {
    EXPECT_DOUBLE_EQ(
        sqrt(0.02),
        statespace.get_distance(SE2::State(1, 1, 1), SE2::State(2, 2, 1)));

    EXPECT_NEAR(
        0.798,
        statespace.get_distance(SE2::State(1, 1, 0), SE2::State(2, 2, 1)),
        0.0001);
}

TEST_F(SE2StatespaceTest, CopyState) {
    SE2::State dest;
    statespace.copy_state(SE2::State(1, 1, 1), &dest);

    EXPECT_EQ(1, dest.X());
    EXPECT_EQ(1, dest.Y());
    EXPECT_EQ(1, dest.Theta());
}

TEST_F(SE2StatespaceTest, GetResolution) {
    double resolution = statespace.get_resolution();
    EXPECT_NEAR(0.1, resolution, 0.0001);
}

TEST_F(SE2StatespaceTest, Add) {
    Eigen::VectorXd a(3);
    a << 1.0, -2.5, 3.4;
    Eigen::VectorXd b(3);
    b << 2.4, 1.4, -7.0;
    Eigen::VectorXd c(3);
    statespace.add(a, b, &c);
    EXPECT_NEAR(3.4, c[0], 0.0001);
    EXPECT_NEAR(-1.1, c[1], 0.0001);
    EXPECT_NEAR(-3.6, c[2], 0.0001);
}

TEST_F(SE2StatespaceTest, AddException) {
    Eigen::VectorXd a(3);
    a << 1.0, -2.5, 3.4;
    Eigen::VectorXd b(2);
    b << 2.4, 1.4;
    Eigen::VectorXd c(3);
    EXPECT_THROW(statespace.add(a, b, &c), std::runtime_error);
}

TEST_F(SE2StatespaceTest, State2Vector) {
    SE2::State dest;
    statespace.copy_state(SE2::State(1, 2, 3), &dest);
    Eigen::VectorXd a = dest.vector();

    EXPECT_EQ(1, a[0]);
    EXPECT_EQ(2, a[1]);
    EXPECT_EQ(3, a[2]);
}

TEST_F(SE2StatespaceTest, Vector2State) {
    Eigen::VectorXd a(3);
    a << 1, 2, 3;
    SE2::State dest;
    dest.from_vector(a);
    EXPECT_EQ(1, dest.X());
    EXPECT_EQ(2, dest.Y());
    EXPECT_EQ(3, dest.Theta());
}

TEST_F(SE2StatespaceTest, Vector2StateException) {
    Eigen::VectorXd a(4);
    a << 1, 2, 3, 0;
    SE2::State dest;
    EXPECT_THROW(dest.from_vector(a), std::runtime_error);
}

}  // namespace test
}  // namespace statespace
}  // namespace libcozmo

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
