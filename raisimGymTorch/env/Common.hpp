//
// Created by jemin on 3/1/22.
//

#ifndef RAISIM_RAISIMGYMTORCH_RAISIMGYMTORCH_ENV_ENVS_COMMON_H_
#define RAISIM_RAISIMGYMTORCH_RAISIMGYMTORCH_ENV_ENVS_COMMON_H_

#include <Eigen/Core>
#include <map>
#include <string>

using Dtype = float;
using EigenRowMajorMat = Eigen::Matrix<Dtype, -1, -1, Eigen::RowMajor>;
using EigenVec = Eigen::Matrix<Dtype, -1, 1>;
using EigenBoolVec = Eigen::Matrix<bool, -1, 1>;
using InfoType = std::map<std::string, std::map<std::string, float>>;

extern int threadCount;

#endif  // RAISIM_RAISIMGYMTORCH_RAISIMGYMTORCH_ENV_ENVS_COMMON_H_
