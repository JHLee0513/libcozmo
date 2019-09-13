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

#include "planner/Dijkstra.hpp"
#include <iostream>

namespace libcozmo {
namespace planner {

    bool Dijkstra::set_start(const int& start_id) {
        if (m_state_space->is_valid_state(
                *m_state_space->get_state(start_id))) {
            m_start_id = start_id;
            return true;    
        }
        return false;
    }

    bool Dijkstra::set_goal(const int& goal_id) {
        if (m_state_space->is_valid_state(
                *m_state_space->get_state(goal_id))) {
            m_goal_id = goal_id;
            return true;
        }
        return false;
    }

    bool Dijkstra::solve(std::vector<int>* actions) {
        
        if (m_goal_id == -1 || m_start_id == -1) {
            return false;
        }
        CostMap costmap;
        ChildToParentMap child_to_parent_map;
        CostMapComparator comparator(costmap);
        std::set<int, CostMapComparator> Q(comparator);
        std::vector<int> path_ids;
        Q.insert(m_start_id);
        costmap[m_start_id] = 0;
        int n = 0;
        int ac = 0;
        while (!Q.empty()) {
            int curr_state = *(Q.begin());
            Q.erase(Q.begin());
            Q.erase(curr_state);
            n++;
            if (curr_state  == m_goal_id ||
                m_se2->get_distance(
                    *m_state_space->get_state(curr_state),
                    *m_state_space->get_state(m_goal_id)) <= m_threshold) {
                extract_path(child_to_parent_map, m_start_id, m_goal_id, actions);
                std::cout << "num expansions: " << n << std::endl;
                std::cout << "actions explored: " << ac << std::endl;
                return true;
            }
            statespace::StateSpace::State* state =
                m_state_space->get_state(curr_state);

            ac = ac + m_action_space->size();
            for (int i = 0; i < m_action_space->size(); i++) {
                actionspace::ActionSpace::Action* action =
                    m_action_space->get_action(i);
                const auto action_ = static_cast<
                    const actionspace::GenericActionSpace::Action*>(action);
                const auto state_ =
                    static_cast<const statespace::SE2::State*>(state);
                model::DeterministicModel::ModelInput input(*action_);
                model::DeterministicModel::ModelOutput* output =
                    static_cast<model::DeterministicModel::ModelOutput*>(
                        m_model->get_prediction(input));
                aikido::statespace::SE2::State s;
                Dijkstra::get_succ(
                    *state_,
                    &s,
                    output->getX(),
                    output->getY(),
                    output->getTheta());
                statespace::SE2::State succ_state;
                m_state_space->continuous_state_to_discrete(s, &succ_state);
                if (m_state_space->is_valid_state(succ_state)) {
                    int id = m_state_space->get_or_create_state(succ_state);
                    double new_cost = costmap[curr_state] +
                        m_state_space->get_distance(*state_, succ_state);
                    if (costmap.find(id) == costmap.end() ||
                        costmap[id] > new_cost) {

                        // std::cout << "SUCCESOR NODE: \n";
                        // std::cout << "ID: " << id << "\n";
                        // std::cout << "Parent: " << curr_state << "\n";
                        // std::cout << "  X: " << succ_state.getX() << "\n";
                        // std::cout << "  Y: " << succ_state.getY() << "\n";
                        // std::cout << "  Theta: " << succ_state.getTheta() << "\n";
                        // std::cout << "  Cost: " << new_cost << "\n";

                        child_to_parent_map[id] = std::make_pair(curr_state, i);
                        costmap[id] = new_cost;
                        assert(Q.find(curr_state) == Q.end());
                        Q.erase(id);
                        Q.insert(id);
                    }
                }
                // add_to_fringe(
                //     &child_to_parent_map;
                //     &costmap,
                //     &Q,
                //     state_
                //     succ_state);
            }
        }
        return false;
    }

    // voidDijkstra::add_to_fringe(
    //     ChildToParentMap* child_to_parent_map;
    //     CostMap* costmap,
    //     std::set<int, CostMapComparator>* Q,
    //     statespace::SE2::State* state_
    //     const statespace::SE2::State& succ_state) {
    //     if (m_state_space->is_valid_state(succ_state)) {
    //         int id = m_state_space->get_or_create_state(succ_state);
    //         double new_cost = *costmap[curr_state] +
    //             m_state_space->get_distance(*state_, succ_state);
    //         if (costmap->find(id) == costmap->end() ||
    //             *costmap[id] > new_cost) {
    //             *child_to_parent_map[id] = std::make_pair(curr_state, i);
    //             *costmap[id] = new_cost;
    //             assert(Q->find(curr_state) == Q->end());
    //             Q->erase(id);
    //             Q->insert(id);
    //         }
    //     }
    // }

    void Dijkstra::get_succ(
        const statespace::SE2::State& state_,
        aikido::statespace::SE2::State* s,
        const double& x,
        const double& y,
        const double& theta) {
        m_state_space->discrete_state_to_continuous(state_, s);
        auto curr_state_isometry = s->getIsometry();
        double x_ = curr_state_isometry.translation()[0] + x;
        double y_ = curr_state_isometry.translation()[1] + y;
        Eigen::Isometry2d t = Eigen::Isometry2d::Identity();
        const Eigen::Rotation2D<double> rot(theta);
        t.linear() = rot.toRotationMatrix();
        t.translation() = Eigen::Vector2d(x_, y_);
        s->setIsometry(t);
    }

    void Dijkstra::extract_path(
        const ChildToParentMap& child_to_parent_map,
        const int& start_id,
        const int& goal_id,
        std::vector<int>* path_actions) {
        if (goal_id == start_id) {
            return;
        }
        auto parent = child_to_parent_map.find(goal_id);
        while (parent != child_to_parent_map.end()) {
            path_actions->push_back(parent->second.second);
            parent = child_to_parent_map.find(parent->second.first);
        }
        std::reverse(path_actions->begin(), path_actions->end());
    }

}  // namespace planner
}  // namespace libcozmo
