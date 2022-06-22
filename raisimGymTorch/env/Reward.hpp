//
// Created by jemin on 20. 9. 22..
//

#ifndef _RAISIM_GYM_TORCH_RAISIMGYMTORCH_ENV_REWARD_HPP_
#define _RAISIM_GYM_TORCH_RAISIMGYMTORCH_ENV_REWARD_HPP_

#include <initializer_list>
#include <string>
#include <map>
#include "Yaml.hpp"

namespace raisim {

struct RewardElement {
    float coefficient = 1.0;
    float reward = 0.0;
    float integral = 0.0;
};

class Reward {
public:
    Reward(std::initializer_list<std::string> names) {
        for (auto& nm : names) {
            rewards_[nm] = raisim::RewardElement();
        }
    }

    Reward() = default;

    void initializeFromConfigurationFile(const Yaml::Node& cfg) {
        for (auto rw = cfg.Begin(); rw != cfg.End(); rw++) {
            rewards_[(*rw).first] = raisim::RewardElement();
            RSFATAL_IF((*rw).second.IsNone() || (*rw).second["coeff"].IsNone(),
                       "Node " + (*rw).first + " or its coefficient doesn't exist");
            rewards_[(*rw).first].coefficient = (*rw).second["coeff"].template As<float>();
        }
    }

    const float& operator[](const std::string& name) {
        return rewards_[name].reward;
    }

    void record(const std::string& name, float reward) {
        RSFATAL_IF(rewards_.find(name) == rewards_.end(),
                   name << " was not found in the configuration file")
        RSISNAN_MSG(reward, name << " is nan")

        rewards_[name].reward = reward * rewards_[name].coefficient;
        rewards_[name].integral += rewards_[name].reward;
    }

    float sum() {
        double rpos = 0;
        double rneg = 0;
        for (auto& rw : rewards_) {
            double value = rw.second.reward;
            if (value >= 0) {
                rpos += value;
            } else {
                rneg += value;
            }
        }
        return rpos * std::exp(0.2 * rneg);
    }

    void reset() {
        for (auto& rw : rewards_) {
            rw.second.integral = 0.f;
            rw.second.reward = 0.f;
        }
    }

    std::map<std::string, float> getStdMap() {
        std::map<std::string, float> rewardMap_;
        float total = 0;
        for (const auto& rw : rewards_) {
            rewardMap_[rw.first] = rw.second.integral;
            total += rw.second.integral;
        }
        rewardMap_["reward_sum"] = total;

        return rewardMap_;
    }

private:
    std::map<std::string, raisim::RewardElement> rewards_;
};

}  // namespace raisim

#endif  //_RAISIM_GYM_TORCH_RAISIMGYMTORCH_ENV_REWARD_HPP_
