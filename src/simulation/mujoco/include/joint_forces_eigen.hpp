#pragma once

#include <mujoco/mujoco.h>
#include <Eigen/Dense>
#include <vector>
#include <utility>
#include <cmath>
#include <stdexcept>

// 使用 Eigen 版的 6D 扳手：M = 力矩, F = 力（都在某个 body 坐标系）
struct WrenchEigen {
    Eigen::Vector3d M;  // [Mx, My, Mz]
    Eigen::Vector3d F;  // [Fx, Fy, Fz]
};

// 分解结果：轴向力 / 剪切力 / 扭矩 / 弯矩
struct DecomposedWrenchEigen {
    double          F_axial_mag;
    Eigen::Vector3d F_axial;

    double          F_shear_mag;
    Eigen::Vector3d F_shear;

    double          M_torsion_mag;
    Eigen::Vector3d M_torsion;

    double          M_bend_mag;
    Eigen::Vector3d M_bend;

    double          M_eq;  // 综合破坏载荷: M_eq = sqrt(M_bend^2 + (0.15 * F_shear)^2)
};

// 每个 body 两端关节受力
struct LinkEndForcesEigen {
    int body_id;    // body index
    int parent_joint;                // 若无父 joint，则为 -1
    WrenchEigen parent_wrench;       // 作用在该 body 上的扳手（body frame）
    std::vector<std::pair<int, WrenchEigen>> child_joints; // (joint id, wrench_on_this_body)
};


//-----------------------------
// 1. 子 body 坐标系下的关节反力
//-----------------------------
inline std::vector<WrenchEigen>
computeJointWrenchesChildBodyEigen(const mjModel* m, mjData* d)
{
    // 确保 internal forces 已根据当前状态计算
    mj_rnePostConstraint(m, d);

    std::vector<WrenchEigen> jw(m->njnt);

    for (int j = 0; j < m->njnt; ++j) {
        int jtype = m->jnt_type[j];
        if (jtype == mjJNT_FREE) {
            jw[j].M.setZero();
            jw[j].F.setZero();
            continue;
        }

        int body = m->jnt_bodyid[j];

        // cfrc_int[body] = [Mx,My,Mz,Fx,Fy,Fz] in body frame at COM
        // 单位说明：MuJoCo使用MKS单位系统（米-千克-秒）
        // 力矩 M 单位：N·m (牛顿·米), 力 F 单位：N (牛顿)
        Eigen::Vector3d tau_body_com(  // 力矩，单位：N·m
            d->cfrc_int[6*body + 0],
            d->cfrc_int[6*body + 1],
            d->cfrc_int[6*body + 2]
        );
        Eigen::Vector3d F_body(  // 力，单位：N
            d->cfrc_int[6*body + 3],
            d->cfrc_int[6*body + 4],
            d->cfrc_int[6*body + 5]
        );

        // body -> world rotation (row-major)
        Eigen::Map<const Eigen::Matrix<double,3,3,Eigen::RowMajor>> R_bw(d->xmat + 9*body);

        // body frame -> world frame
        Eigen::Vector3d F_world        = R_bw * F_body;
        Eigen::Vector3d tau_world_com  = R_bw * tau_body_com;

        // COM 和 关节位置（world）
        Eigen::Vector3d p_com(
            d->xipos[3*body + 0],
            d->xipos[3*body + 1],
            d->xipos[3*body + 2]
        );
        Eigen::Vector3d p_joint(
            d->xanchor[3*j + 0],
            d->xanchor[3*j + 1],
            d->xanchor[3*j + 2]
        );

        // r = COM - joint (单位：m)
        Eigen::Vector3d r = p_com - p_joint;

        // tau_joint(world) = tau_com + r × F
        // 单位：N·m = N·m + m × N
        // 注意：如果r很大（COM距离关节较远），r×F会产生较大的力矩
        // 例如：r=0.1m, F=100N → r×F = 10 N·m
        Eigen::Vector3d tau_world_joint = tau_world_com + r.cross(F_world);  // 单位：N·m
        Eigen::Vector3d F_world_joint   = F_world;  // 单位：N

        // world -> body (child) frame
        Eigen::Matrix3d R_wb = R_bw.transpose();
        Eigen::Vector3d tau_body_joint = R_wb * tau_world_joint;  // 单位：N·m
        Eigen::Vector3d F_body_joint   = R_wb * F_world_joint;    // 单位：N

        jw[j].M = tau_body_joint;  // 单位：N·m
        jw[j].F = F_body_joint;    // 单位：N
    }

    return jw;
}


//---------------------------------------------
// 2. 按关节轴分解扳手：轴向力 / 剪切力 / 扭矩 / 弯矩
//---------------------------------------------
inline DecomposedWrenchEigen
decomposeWrenchBodyFrameEigen(const WrenchEigen& w,
                              const Eigen::Vector3d& axis_body)
{
    DecomposedWrenchEigen out;

    Eigen::Vector3d tau = w.M;
    Eigen::Vector3d F   = w.F;

    double norm_a = axis_body.norm();
    if (norm_a < 1e-8) {
        throw std::runtime_error("joint axis is near zero in decomposeWrenchBodyFrameEigen");
    }
    Eigen::Vector3d a_hat = axis_body / norm_a;

    // 轴向力
    out.F_axial_mag = F.dot(a_hat);
    out.F_axial     = out.F_axial_mag * a_hat;

    // 剪切力
    out.F_shear     = F - out.F_axial;
    out.F_shear_mag = out.F_shear.norm();

    // 扭矩
    out.M_torsion_mag = tau.dot(a_hat);
    out.M_torsion     = out.M_torsion_mag * a_hat;

    // 弯矩
    out.M_bend     = tau - out.M_torsion;
    out.M_bend_mag = out.M_bend.norm();

    // 综合破坏载荷: M_eq = sqrt(M_bend^2 + (0.15 * F_shear)^2)
    // 注：0.15 是 0.1-0.2 范围的中间值，可根据需要调整
    double coefficient = 0.15;  // 可以改为 0.1 或 0.2
    out.M_eq = std::sqrt(out.M_bend_mag * out.M_bend_mag + 
                         (coefficient * out.F_shear_mag) * (coefficient * out.F_shear_mag));

    return out;
}


//---------------------------------------------------------
// 3. 作用在父 body 上、在父 body 坐标系下的关节反力
//---------------------------------------------------------
inline std::vector<WrenchEigen>
computeJointWrenchesParentBodyEigen(const mjModel* m,
                                    const mjData* d,
                                    const std::vector<WrenchEigen>& childWrenches)
{
    std::vector<WrenchEigen> parentWrenches(m->njnt);

    for (int j = 0; j < m->njnt; ++j) {
        int jtype = m->jnt_type[j];
        if (jtype == mjJNT_FREE) {
            parentWrenches[j].M.setZero();
            parentWrenches[j].F.setZero();
            continue;
        }

        int child  = m->jnt_bodyid[j];
        int parent = m->body_parentid[child];

        if (parent < 0) {
            parentWrenches[j].M.setZero();
            parentWrenches[j].F.setZero();
            continue;
        }

        const WrenchEigen& wc = childWrenches[j];

        // child -> world
        Eigen::Map<const Eigen::Matrix<double,3,3,Eigen::RowMajor>> R_cbw(d->xmat + 9*child);

        Eigen::Vector3d tau_world = R_cbw * wc.M;
        Eigen::Vector3d F_world   = R_cbw * wc.F;

        // 作用在 parent 上是反号
        Eigen::Vector3d tau_world_parent = -tau_world;
        Eigen::Vector3d F_world_parent   = -F_world;

        // parent -> world
        Eigen::Map<const Eigen::Matrix<double,3,3,Eigen::RowMajor>> R_pbw(d->xmat + 9*parent);
        Eigen::Matrix3d R_wpb = R_pbw.transpose();

        parentWrenches[j].M = R_wpb * tau_world_parent;
        parentWrenches[j].F = R_wpb * F_world_parent;
    }

    return parentWrenches;
}


//--------------------------------------
// 4. 每个 link 两端关节受力汇总
//--------------------------------------
inline std::vector<LinkEndForcesEigen>
collectLinkEndWrenchesEigen(const mjModel* m,
                            const std::vector<WrenchEigen>& jointWrenchesChild,
                            const std::vector<WrenchEigen>& jointWrenchesParent)
{
    std::vector<LinkEndForcesEigen> links(m->nbody);

    for (int b = 0; b < m->nbody; ++b) {
        links[b].body_id      = b;
        links[b].parent_joint = -1;
        links[b].parent_wrench.M.setZero();
        links[b].parent_wrench.F.setZero();
        links[b].child_joints.clear();
    }

    for (int j = 0; j < m->njnt; ++j) {
        int jtype = m->jnt_type[j];
        if (jtype == mjJNT_FREE) {
            continue;
        }

        int child  = m->jnt_bodyid[j];
        int parent = m->body_parentid[child];

        // 对子 body：这个 joint 是“靠 parent 端”的关节
        if (child >= 0) {
            links[child].parent_joint  = j;
            links[child].parent_wrench = jointWrenchesChild[j];
        }

        // 对父 body：这个 joint 是“靠子端”的一个关节
        if (parent >= 0) {
            links[parent].child_joints.emplace_back(j, jointWrenchesParent[j]);
        }
    }

    return links;
}


//--------------------------------------
// 5. 使用示例（可删）
//--------------------------------------
// 放在 .cpp 里测试用；集成到工程后可以去掉这段。
/*
#include <iostream>

inline void exampleUsage()
{
    char error[1000];
    mjModel* m = mj_loadXML("serial_pm_v2_mesh.xml", nullptr, error, 1000);
    if (!m) {
        std::cerr << "Load model failed: " << error << std::endl;
        return;
    }
    mjData* d = mj_makeData(m);

    // 初始前向
    mj_forward(m, d);

    // 1) 每个关节在子 link 坐标系下的 6D 反力
    auto jointChild = computeJointWrenchesChildBodyEigen(m, d);

    // 2) 每个关节在父 link 坐标系下的 6D 反力
    auto jointParent = computeJointWrenchesParentBodyEigen(m, d, jointChild);

    // 3) 每个 body 两端关节受力
    auto linkEnds = collectLinkEndWrenchesEigen(m, jointChild, jointParent);

    // 4) 例子：某个关节的载荷分解
    int j_knee = mj_name2id(m, mjOBJ_JOINT, "J03_KNEE_PITCH_L");
    if (j_knee >= 0) {
        const WrenchEigen& w_knee = jointChild[j_knee];
        Eigen::Vector3d axis = Eigen::Vector3d(
            m->jnt_axis[3*j_knee + 0],
            m->jnt_axis[3*j_knee + 1],
            m->jnt_axis[3*j_knee + 2]
        );
        auto comp = decomposeWrenchBodyFrameEigen(w_knee, axis);
        std::cout << "knee axial F = " << comp.F_axial_mag
                  << ", shear F = " << comp.F_shear_mag
                  << ", bend M = " << comp.M_bend_mag
                  << ", torsion M = " << comp.M_torsion_mag
                  << std::endl;
    }

    mj_deleteData(d);
    mj_deleteModel(m);
}
*/

