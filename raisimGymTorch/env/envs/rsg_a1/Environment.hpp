//----------------------------//
// This file is part of RaiSim//
// Copyright 2020, RaiSim Tech//
//----------------------------//

#pragma once

#include <stdlib.h>
#include <set>
#include <algorithm>
#include <cmath>
#include "../../RaisimGymEnv.hpp"

/* Convention
 *
 *   observation space = [ height,                       n =  1, si (start index) =  0
 *                         body roll,,                   n =  1, si =  1
 *                         body pitch,,                  n =  1, si =  2
 *                         joint angles,                 n = 12, si =  3
 *                         body Linear velocities,       n =  3, si = 15
 *                         body Angular velocities,      n =  3, si = 18
 *                         joint velocities,             n = 12, si = 21
 *                         contacts binary vector,       n =  4, si = 33
 *                         previous action,              n = 12, si = 37 ] total 49
 *
 *   action space      = [ joint angles                  n = 12, si =  0 ] total 12
 */

namespace raisim {

class ENVIRONMENT : public RaisimGymEnv {

public:
    explicit ENVIRONMENT(const std::string& resourceDir, const Yaml::Node& cfg, bool visualizable)
        : RaisimGymEnv(resourceDir, cfg), visualizable_(visualizable) {

        /// create world
        world_ = std::make_unique<raisim::World>();

        // Load configuration
        setSimulationTimeStep(cfg["simulation_dt"].template As<double>());
        setControlTimeStep(cfg["control_dt"].template As<double>());
        k_c = cfg["k_0"].template As<double>();
        k_d = cfg["k_d"].template As<double>();
        // ...

        /// add objects
        a1_ = world_->addArticulatedSystem(resourceDir_ + "/a1/urdf/a1.urdf");
        a1_->setName("Unitree A1");
        a1_->setControlMode(raisim::ControlMode::PD_PLUS_FEEDFORWARD_TORQUE);

        // Terrain
        // auto ground = world_->addGround(); // Flat terrain
        raisim::TerrainProperties terrainProperties;  // Randomized terrain
        terrainProperties.frequency = cfg["terrain"]["frequency"].template As<double>();
        terrainProperties.zScale = cfg["terrain"]["zScale"].template As<double>();
        terrainProperties.xSize = cfg["terrain"]["xSize"].template As<double>();
        terrainProperties.ySize = cfg["terrain"]["ySize"].template As<double>();
        terrainProperties.xSamples = cfg["terrain"]["xSamples"].template As<size_t>();
        terrainProperties.ySamples = cfg["terrain"]["ySamples"].template As<size_t>();
        terrainProperties.fractalOctaves = cfg["terrain"]["fractalOctaves"].template As<size_t>();
        terrainProperties.fractalLacunarity =
            cfg["terrain"]["fractalLacunarity"].template As<double>();
        terrainProperties.fractalGain = cfg["terrain"]["fractalGain"].template As<double>();

        auto hm = world_->addHeightMap(0.0, 0.0, terrainProperties);

        /// get robot data
        gcDim_ = a1_->getGeneralizedCoordinateDim();
        gvDim_ = a1_->getDOF();
        nJoints_ = gvDim_ - 6;

        /// initialize containers
        gc_.setZero(gcDim_);
        gc_init_.setZero(gcDim_);
        gv_.setZero(gvDim_);
        gv_init_.setZero(gvDim_);
        pTarget_.setZero(gcDim_);
        vTarget_.setZero(gvDim_);

        // Generate robot initial pose
        x0Dist_ = std::uniform_real_distribution<double>(-1, 1);
        y0Dist_ = std::uniform_real_distribution<double>(-1, 1);

        /// this is nominal standing configuration of unitree A1
        // P_x, P_y, P_z, 1.0, A_x, A_y, A_z, FR_hip, FR_thigh, FR_calf, FL_hip, FL_thigh, FL_calf,
        // RR_hip, RR_thigh, RR_calf, RL_hip, RL_thigh, RL_calf. gc_init_ << 0.0, 0.0, 0.39, 1.0,
        // 0.0, 0.0, 0.0, 0.06, 0.6, -1.2, -0.06, 0.6, -1.2, 0.06, 0.6, -1.2, -0.06, 0.6, -1.2;
        gc_init_ << 0.0, 0.0, 0.45, 1.0, 0.0, 0.0, 0.0,
            0.06, 0.6, -1.2, -0.06, 0.6, -1.2, 0.06, 0.6, -1.2, -0.06, 0.6, -1.2;

        /// set pd gains
        Eigen::VectorXd jointPgain(gvDim_), jointDgain(gvDim_);
        jointPgain.setZero();
        jointPgain.tail(nJoints_).setConstant(55.0);
        jointDgain.setZero();
        jointDgain.tail(nJoints_).setConstant(0.8);
        a1_->setPdGains(jointPgain, jointDgain);
        a1_->setGeneralizedForce(Eigen::VectorXd::Zero(gvDim_));

        /// MUST BE DONE FOR ALL ENVIRONMENTS
        obDim_ = 50;  /// convention described on top
        actionDim_ = nJoints_;
        actionMean_.setZero(actionDim_);
        actionStd_.setZero(actionDim_);
        obDouble_.setZero(obDim_);
        obMean_.setZero(obDim_);
        obStd_.setZero(obDim_);

        /// action scaling
        actionMean_ = gc_init_.tail(nJoints_);
        actionStd_.setConstant(0.3);

        // Observation scaling
        obMean_ << 0.37,                         // height
            Eigen::VectorXd::Constant(2, 0.0),   // body roll & pitch
            gc_init_.tail(nJoints_),             // joint nominal angles
            Eigen::VectorXd::Constant(6, 0.0),   // body linear & angular velocity
            Eigen::VectorXd::Constant(12, 0.0),  // joint velocity
            Eigen::VectorXd::Constant(4, 0.0),   // contacts binary vector
            Eigen::VectorXd::Constant(12, 0.0),  // previous action
            0.6;

        obStd_ << 0.01,                          // height
            Eigen::VectorXd::Constant(2, 1.0),   // body roll & pitch
            Eigen::VectorXd::Constant(12, 1.0),  // joint nominal angles
            1.0 / 1.5, 1.0 / 0.5, 1.0 / 0.5,     // body linear velocity
            1.0 / 2.5, 1.0 / 2.5, 1.0 / 2.5,     // body angular velocity
            Eigen::VectorXd::Constant(12, .01),  // joint velocity
            Eigen::VectorXd::Constant(4, 1.0),   // contacts binary vector
            Eigen::VectorXd::Constant(12, 1.0),  // previous action
            0.28;


        groundImpactForces_.setZero();
        previousGroundImpactForces_.setZero();
        previousJointPositions_.setZero(nJoints_);
        previous2JointPositions_.setZero(nJoints_);
        previousTorque_ = a1_->getGeneralizedForce().e().tail(nJoints_);

        /// indices of links that should make contact with ground
        contactIndices_.insert(a1_->getBodyIdx("FL_calf"));
        contactIndices_.insert(a1_->getBodyIdx("FR_calf"));
        contactIndices_.insert(a1_->getBodyIdx("RL_calf"));
        contactIndices_.insert(a1_->getBodyIdx("RR_calf"));

        // Define mapping of body ids to sequential indices
        contactSequentialIndex_[a1_->getBodyIdx("FL_calf")] = 0;
        contactSequentialIndex_[a1_->getBodyIdx("FR_calf")] = 1;
        contactSequentialIndex_[a1_->getBodyIdx("RL_calf")] = 2;
        contactSequentialIndex_[a1_->getBodyIdx("RR_calf")] = 3;

        // Initialize materials
        world_->setMaterialPairProp("default", "rubber", 0.8, 0.15, 0.001);

        command_ << 1.0, 0.0, 0.0;

        // TODO: Move values to config
        // Initialize environmental sampler distributions
        decisionDist_ = std::uniform_real_distribution<double>(0, 1);
        frictionDist_ = std::uniform_real_distribution<double>(0.5, 1.25);
        // frictionDist_ = std::uniform_real_distribution<double>(0.7, 3.5);
        kpDist_ = std::uniform_real_distribution<double>(50, 60);
        kdDist_ = std::uniform_real_distribution<double>(0.4, 0.8);
        comDist_ = std::uniform_real_distribution<double>(-0.0015, 0.0015);
        motorStrengthDist_ = std::uniform_real_distribution<double>(0.9, 1.1);
        speedDist_ = std::uniform_real_distribution<double>(
            cfg["target_speed"]["low"].template As<double>(),
            cfg["target_speed"]["up"].template As<double>()
        );

        initialActuationUpperLimits_ = a1_->getActuationUpperLimits().e().tail(nJoints_);
        initialActuationLowerLimits_ = a1_->getActuationLowerLimits().e().tail(nJoints_);

        /// Reward coefficients
        rewards_.initializeFromConfigurationFile(cfg["reward"]);

        /// visualize if it is the first environment
        if (visualizable_) {
            server_ = std::make_unique<raisim::RaisimServer>(world_.get());
            server_->launchServer();
            server_->focusOn(a1_);
        }
    }

    virtual void init() override {
    }

    virtual void reset() override {
        // std::cout << "env.reset" << std::endl;
        resampleEnvironmentalParameters();
        gc_init_[0] = x0Dist_(randomGenerator_);
        gc_init_[1] = y0Dist_(randomGenerator_);

        a1_->setState(gc_init_, gv_init_);

        previousJointPositions_ = gc_.tail(nJoints_);
        previous2JointPositions_ = gc_.tail(nJoints_);
        updateObservation();
        steps_ = 0;

        // for (const auto& [name, value] : rewards_.getStdMap()) {
        //     std::cout << name << " " << value << std::endl;
        // }
        // std::cout << "----------\n\n";

        rewards_.reset();
        targetSpeed_ = speedDist_(randomGenerator_);
    }

    virtual void curriculumUpdate() override {
        k_c = std::min(pow(k_c, k_d), 1.0);
    }

    virtual float step(const Eigen::Ref<EigenVec>& action) override {
        /// action scaling
        Eigen::VectorXd pTarget12 = action.cast<double>();
        // pTarget12 = pTarget12.cwiseMin(1).cwiseMax(-1);
        pTarget12 = actionMean_ + pTarget12.cwiseProduct(actionStd_);
        pTarget_.tail(nJoints_) = pTarget12;

        a1_->setPdTarget(pTarget_, vTarget_);

        for (int i = 0; i < int(control_dt_ / simulation_dt_ + 1e-10); i++) {
            if (server_) {
                server_->lockVisualizationServerMutex();
            }
            world_->integrate();
            if (server_) {
                server_->unlockVisualizationServerMutex();
            }
        }

        // Record values for next step calculations
        previousTorque_ = a1_->getGeneralizedForce().e().tail(nJoints_);
        previous2JointPositions_ = previousJointPositions_;
        previousJointPositions_ = gc_.tail(nJoints_);
        previousGroundImpactForces_ = groundImpactForces_;

        updateObservation();

        rewards_.record("BaseForwardVelocity", 0.6 * calculateBaseForwardVelocityCost());
        rewards_.record("BaseLateralAndRotation", -calculateBaseLateralAndRotationCost());
        rewards_.record("BaseHeight", -0.25 * calculateBaseHeightCost());
        rewards_.record("Torque", -0.25 * calculateTorqueCost());
        rewards_.record("JointSpeed", -0.25 * 0.36 * calculateJointSpeedCost());
        rewards_.record("AirTime", calculateAirTimeCost());
        rewards_.record("Slip", -0.25 * 0.36 * calculateSlipCost());
        rewards_.record("Orientation", -0.25 * calculateOrientationCost());
        rewards_.record("Smoothness", -0.25 * calculateSmoothnessCost());

        // New terms
        rewards_.record("Work", -0.25 * calculateWorkCost());
        rewards_.record("GroundImpact", -0.25 * 0.36 * calculateGroundImpactCost());
        // rewards_.record("ActionMagnitude", -calculateActionMagnitudeCost());
        rewards_.record("ZAcceleration", -calculateZAccelerationCost());

        // Apply random force to the COM
        auto applyingForceDecision = decisionDist_(randomGenerator_);
        if (applyingForceDecision < 0.5) {
            auto externalEffort = 1000 * Eigen::VectorXd::Random(3);
            a1_->setExternalForce(a1_->getBodyIdx("base"), externalEffort);
        }

        // Apply random torque to the COM
        applyingForceDecision = decisionDist_(randomGenerator_);
        if (applyingForceDecision < 0.05) {
            auto externalTorque = 100 * Eigen::VectorXd::Random(3);
            a1_->setExternalTorque(a1_->getBodyIdx("base"), externalTorque);
        }

        ++steps_;
        return rewards_.sum();
    }

    virtual InfoType getInfo() override {
        return {
            {"reward", rewards_.getStdMap()},
            {"stats", {
                {"speedx", bodyLinearVel_[0]},
                {"speedy", bodyLinearVel_[1]},
                {"posx", gc_[0]},
                {"posy", gc_[1]},
                {"k_c", k_c}
            }}
        };
    }

    void updateObservation() {
        a1_->getState(gc_, gv_);
        raisim::Vec<4> quat = {gc_[3], gc_[4], gc_[5], gc_[6]};
        raisim::Mat<3, 3> rot;
        raisim::quatToRotMat(quat, rot);
        bodyLinearVel_ = rot.e().transpose() * gv_.segment(0, 3);
        bodyAngularVel_ = rot.e().transpose() * gv_.segment(3, 3);

        Eigen::VectorXd contacts;
        contacts.setZero(footContactState_.size());
        for (auto& fs : footContactState_) {
            fs = false;
        }
        groundImpactForces_.setZero();

        /// Dirive contacts vector
        for (auto& contact : a1_->getContacts()) {
            if (!contact.isSelfCollision() && contact.getPairObjectBodyType() == BodyType::STATIC) {
                if (contactIndices_.find(contact.getlocalBodyIndex()) != contactIndices_.end()) {
                    auto groundImpactForce = contact.getImpulse().e().norm() / world_->getTimeStep();
                    auto bodyIndex = contact.getlocalBodyIndex();
                    groundImpactForces_[contactSequentialIndex_[bodyIndex]] = groundImpactForce;
                    footContactState_[contactSequentialIndex_[bodyIndex]] = true;
                    contacts[contactSequentialIndex_[bodyIndex]] = 1;
                }
            }
        }

        // Nosify observations
        auto velocitiesNoised = gv_.tail(nJoints_) + 0.5 * Eigen::VectorXd::Random(nJoints_);

        auto bodyLinearVelocityNoised =
            bodyLinearVel_ + 0.08 * Eigen::VectorXd::Random(bodyLinearVel_.size());
        auto bodyAngularVelocityNoised =
            bodyAngularVel_ + 0.16 * Eigen::VectorXd::Random(bodyAngularVel_.size());

        double euler_angles[3];
        raisim::quatToEulerVec(&gc_[3], euler_angles);

        obDouble_ << gc_[2],                   // body height 1
            euler_angles[0], euler_angles[1],  // body roll & pitch 2
            gc_.tail(nJoints_),                // joint angles 12
            bodyLinearVelocityNoised,          // body linear 3
            bodyAngularVelocityNoised,         // angular velocity 3
            velocitiesNoised,                  // joint velocity 12
            contacts,                          // contacts binary vector 4
            previousJointPositions_,           // previous action 12
            targetSpeed_;
    }

    virtual void observe(Eigen::Ref<EigenVec> ob) override {
        /// convert it to float
        ob = (obDouble_ - obMean_).cwiseProduct(obStd_).cast<float>();
    }

    virtual bool isTerminalState(float& terminalReward) override {
        // Terminal condition
        double euler_angles[3];
        raisim::quatToEulerVec(&gc_[3], euler_angles);
        if (gc_[2] < 0.28 || fabs(euler_angles[0]) > 0.4 || fabs(euler_angles[1]) > 0.2) {
            terminalReward = float(terminalRewardCoeff_);
            return true;
        }

        if (steps_ == maxSteps_) {
            terminalReward = 0;
            return true;
        }

        terminalReward = 0;
        return false;
    }

private:
    int gcDim_, gvDim_, nJoints_;
    bool visualizable_ = false;
    raisim::ArticulatedSystem* a1_;
    Eigen::VectorXd gc_init_, gv_init_, gc_, gv_, pTarget_, vTarget_;
    double terminalRewardCoeff_ = -10.;
    Eigen::VectorXd actionMean_, actionStd_, obDouble_, obMean_, obStd_;
    Eigen::Vector3d bodyLinearVel_, bodyAngularVel_;
    Eigen::Vector4d groundImpactForces_;
    Eigen::VectorXd previousJointPositions_;
    Eigen::VectorXd previous2JointPositions_;
    Eigen::VectorXd previousTorque_;
    Eigen::Vector4d previousGroundImpactForces_;

    // Curriculum factors
    double k_c, k_d;
    double targetSpeed_ = 0.6;

    Eigen::Vector3d command_;

    std::random_device randomGenerator_;
    std::uniform_real_distribution<double> x0Dist_;
    std::uniform_real_distribution<double> y0Dist_;

    // Random stuff for environmental parameters
    std::uniform_real_distribution<double> decisionDist_;
    std::uniform_real_distribution<double> frictionDist_;
    std::uniform_real_distribution<double> kpDist_;
    std::uniform_real_distribution<double> kdDist_;
    std::uniform_real_distribution<double> comDist_;
    std::uniform_real_distribution<double> motorStrengthDist_;

    std::uniform_real_distribution<double> speedDist_;

    Eigen::VectorXd initialActuationUpperLimits_;
    Eigen::VectorXd initialActuationLowerLimits_;

    // Contacts information
    std::set<size_t> contactIndices_;
    std::array<bool, 4> footContactState_;
    std::unordered_map<int, int> contactSequentialIndex_;

    int maxSteps_ = 3500;
    int steps_ = 0;

private:
    void resampleEnvironmentalParameters() {
        // std::cout << "Resampling enviroment parameters: " << std::endl;

        // Center Of Mass
        auto& baseCOM = a1_->getBodyCOM_B().at(0);
        baseCOM[0] = k_c * comDist_(randomGenerator_);
        baseCOM[1] = k_c * comDist_(randomGenerator_);
        baseCOM[2] = k_c * comDist_(randomGenerator_);
        a1_->updateMassInfo();

        // std::cout << "\nCOM: " << baseCOM.e().transpose() << std::endl;

        // Friction
        auto frictionMean = (frictionDist_.a() + frictionDist_.b()) / 2.;
        auto friction = frictionMean + k_c * (frictionDist_(randomGenerator_) - frictionMean);
        world_->setDefaultMaterial(friction, 0, 0);
        world_->setMaterialPairProp("default", "rubber", friction, 0.15, 0.001);

        // if (visualizable_) std::cout << "\nFriction: " << friction << std::endl;

        // P and D gains
        auto KpMean = (kpDist_.a() + kpDist_.b()) / 2.;
        auto Kp = KpMean + k_c * (kpDist_(randomGenerator_) - KpMean);
        auto Kd = kdDist_.a() + k_c * (kdDist_(randomGenerator_) - kdDist_.a());

        Eigen::VectorXd jointPgain(gvDim_), jointDgain(gvDim_);
        jointPgain.setZero();
        jointPgain.tail(nJoints_).setConstant(Kp);
        jointDgain.setZero();
        jointDgain.tail(nJoints_).setConstant(Kd);
        a1_->setPdGains(jointPgain, jointDgain);

        // std::cout << "\nPgain: " << jointPgain.transpose() << std::endl;
        // std::cout << "\nDgain: " << jointDgain.transpose() << std::endl;

        // Motors strength
        Eigen::VectorXd actuationUpperLimits, actuationLowerLimits;
        actuationUpperLimits.setZero(gvDim_);
        actuationLowerLimits.setZero(gvDim_);

        auto motorStrengthMean = (motorStrengthDist_.a() + motorStrengthDist_.b()) / 2.;
        auto motorStrength =
            motorStrengthMean + k_c * (motorStrengthDist_(randomGenerator_) - motorStrengthMean);
        actuationUpperLimits.tail(nJoints_) = motorStrength * initialActuationUpperLimits_;
        actuationLowerLimits.tail(nJoints_) = motorStrength * initialActuationLowerLimits_;
        a1_->setActuationLimits(actuationUpperLimits, actuationLowerLimits);

        // std::cout << "\nActuationUpperLimits: " << actuationUpperLimits.transpose() << std::endl;
        // std::cout << "\nActuationLowerLimits: " << actuationLowerLimits.transpose() << std::endl;
    }

    //
    // Cost terms calculations
    //

    inline double calculateBaseForwardVelocityCost() {
        return std::exp(-std::pow(bodyLinearVel_[0] - targetSpeed_, 2.0) / 0.1);
    }

    inline double calculateBaseLateralAndRotationCost() {
        return k_c *
               (bodyLinearVel_[1] * bodyLinearVel_[1] + bodyAngularVel_[2] * bodyAngularVel_[2]);
    }

    inline double calculateBaseHeightCost() {
        return k_c * (gc_init_[2] - gc_[2]) * (gc_init_[2] - gc_[2]);
    }

    inline double calculateTorqueCost() {
        return k_c * a1_->getGeneralizedForce().e().tail(nJoints_).squaredNorm();
    }

    inline double calculateJointSpeedCost() {
        auto joint_velocities = gv_.tail(nJoints_);
        const double speedCoef = std::max(std::abs(targetSpeed_), 0.6);
        return k_c * joint_velocities.squaredNorm() / (speedCoef * speedCoef);
    }

    inline double calculateAirTimeCost() {
        auto p_f_hat = cfg_["reward"]["AirTime"]["desired_foot_height"].template As<double>();

        double footAirTimeCost = 0.0;
        for (auto footBodyIndex : contactIndices_) {
            raisim::Vec<3> pos, vel;
            a1_->getPosition(footBodyIndex, pos);
            a1_->getVelocity(footBodyIndex, vel);

            // We only use xy velocity components
            vel[2] = 0.0;

            if (!footContactState_[contactSequentialIndex_[footBodyIndex]]) {
                footAirTimeCost += (p_f_hat - pos[2]) * (p_f_hat - pos[2]) * vel.squaredNorm();
            }
        }

        return k_c * footAirTimeCost;
    }

    inline double calculateSlipCost() {
        double footSlipCost = 0.0;
        for (auto footBodyIndex : contactIndices_) {
            raisim::Vec<3> vel;
            a1_->getVelocity(footBodyIndex, vel);

            // We only use xy velocity components
            vel[2] = 0.0;

            if (footContactState_[contactSequentialIndex_[footBodyIndex]]) {
                footSlipCost += vel.squaredNorm();
            }
        }

        const double speedCoef = std::max(std::abs(targetSpeed_), 0.6);
        return k_c * footSlipCost / (speedCoef * speedCoef);
    }

    inline double calculateOrientationCost() {
        auto angles_roll_pitch = gc_.segment(4, 2);
        return k_c * angles_roll_pitch.squaredNorm();
    }

    inline double calculateSmoothnessCost() {
        auto torque = a1_->getGeneralizedForce().e().tail(nJoints_);
        return k_c * (previousTorque_ - torque).squaredNorm();
    }

    //
    // New reward terms
    //
    inline double calculateWorkCost() {
        auto torque = a1_->getGeneralizedForce().e().tail(nJoints_);
        auto jointPositions = gc_.tail(nJoints_);
        return k_c * fabs(torque.transpose() * (jointPositions - previousJointPositions_));
    }

    inline double calculateGroundImpactCost() {
        const double speedCoef = std::max(std::abs(targetSpeed_), 0.6);
        return k_c * (groundImpactForces_ - previousGroundImpactForces_).squaredNorm() / (speedCoef * speedCoef);
    }

    inline double calculateActionMagnitudeCost() {
        auto jointPositions = gc_.tail(nJoints_);
        return k_c * jointPositions.squaredNorm();
    }

    inline double calculateZAccelerationCost() {
        return k_c * (bodyLinearVel_[2] * bodyLinearVel_[2]);
    }
};

}  // namespace raisim
