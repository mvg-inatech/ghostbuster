#ifndef TOOLS_HPP
#define TOOLS_HPP

#include <Eigen/Core>
#include <unordered_map>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#define HASH_P 116101
#define MAX_N 10000000000
#define SKEW_SYM_MATRX(v) 0.0,-v[2],v[1],v[2],0.0,-v[0],-v[1],v[0],0.0
#define PLM(a) vector<Eigen::Matrix<double, a, a>, Eigen::aligned_allocator<Eigen::Matrix<double, a, a>>>
#define PLV(a) vector<Eigen::Matrix<double, a, 1>, Eigen::aligned_allocator<Eigen::Matrix<double, a, 1>>>

#define G_m_s2 9.8
#define DIM 15
#define NMATCH 5


struct PointXYZIRGB
{
    PCL_ADD_POINT4D;                // x, y, z, padding
    float intensity;
    union
    {
        struct
        {
            uint8_t b;
            uint8_t g;
            uint8_t r;
            uint8_t a;
        };
        std::uint32_t rgb;
    };
    float normal_x;
    float normal_y;
    float normal_z;
    float curvature;      // scratch / pv.var trace (saved as "pv_var" in PCD)
    float balm_res;       // post-BA point-to-plane distance (m); -1 = not on a feature plane
    float ekf_res;        // EKF odometry point-to-plane residual (m); -1 = no match
    float cell_id;        // fixed finest-grid cell ID (cast from int32); -1 = unknown
    float reflectivity;      // Ouster calibrated reflectivity (0-255); 0 for non-Ouster sensors
    float plane_quality;     // λ_min/λ_max of the voxel plane; low≈flat, -1 = no plane
    float obs_count;         // unique scans that contributed to this point's octree leaf
    float view_diversity;    // λ_min of the MEAN viewing-direction scatter (0=one dir, 1/3=isotropic)
    float temp_consistency;  // std-dev of point-to-plane residuals over time (-1 = no plane yet)
    float ambient;           // Ouster background NIR ("near_ir"); scene illumination, independent of the return
    float ring;              // Ouster beam index 0-127; also encodes elevation angle
    float scan_time;         // seconds since the start of this sweep (Ouster per-point `t`)
    float range_detach;      // |range - one-sided extrapolation| on the raw
                             // range image, SMALLER side, normalised by range
    float range_edge;        // same, LARGER side: size of the depth discontinuity
    float int_detach;        // photometric version, on intensity / intensity
    float ring_rough;        // spread of per-column range increments / range;
                             // local roughness ALONG the ring, ~0 on any smooth
                             // surface however steeply it is grazed
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
} EIGEN_ALIGN16;

POINT_CLOUD_REGISTER_POINT_STRUCT(PointXYZIRGB,
    (float, x, x)
    (float, y, y)
    (float, z, z)
    (float, intensity, intensity)
    (std::uint32_t, rgb, rgb)
    (float, normal_x, normal_x)
    (float, normal_y, normal_y)
    (float, normal_z, normal_z)
    (float, curvature, pv_var)
    (float, balm_res, balm_res)
    (float, ekf_res, ekf_res)
    (float, cell_id, cell_id)
    (float, reflectivity, reflectivity)
    (float, plane_quality, plane_quality)
    (float, obs_count, obs_count)
    (float, view_diversity, view_diversity)
    (float, temp_consistency, temp_consistency)
    (float, ambient, ambient)
    (float, ring, ring)
    (float, scan_time, scan_time)
    (float, range_detach, range_detach)
    (float, range_edge, range_edge)
    (float, int_detach, int_detach)
    (float, ring_rough, ring_rough)
)

struct SaveFieldFlags {
  bool pv_var          = true;
  bool balm_res        = true;
  bool ekf_res         = true;
  bool cell_id         = true;
  bool reflectivity    = true;
  bool plane_quality   = true;
  bool obs_count       = true;
  bool view_diversity  = true;
  bool temp_consistency = true;
  bool ambient         = true;
  bool ring            = true;
  bool scan_time       = true;
  bool range_detach    = true;
  bool range_edge      = true;
  bool int_detach      = true;
  bool ring_rough      = true;
};

typedef PointXYZIRGB PointType;
using namespace std;

Eigen::Matrix3d I33(Eigen::Matrix3d::Identity());

class VOXEL_LOC
{
public:
  int64_t x, y, z;

  VOXEL_LOC(int64_t vx=0, int64_t vy=0, int64_t vz=0): x(vx), y(vy), z(vz){}

  bool operator == (const VOXEL_LOC &other) const
  {
    return (x==other.x && y==other.y && z==other.z);
  }
};

namespace std
{
  template<>
  struct hash<VOXEL_LOC>
  {
    size_t operator() (const VOXEL_LOC &s) const
    {
      using std::size_t; using std::hash;
      // return ((hash<int64_t>()(s.x) ^ (hash<int64_t>()(s.y) << 1)) >> 1) ^ (hash<int64_t>()(s.z) << 1);
      return (((hash<int64_t>()(s.z)*HASH_P)%MAX_N + hash<int64_t>()(s.y))*HASH_P)%MAX_N + hash<int64_t>()(s.x);
    }
  };
}

Eigen::Matrix3d Exp(const Eigen::Vector3d &ang)
{
  double ang_norm = ang.norm();
  // if (ang_norm > 0.00001)
  if (ang_norm >= 1e-11)
  {
    Eigen::Vector3d r_axis = ang / ang_norm;
    Eigen::Matrix3d K;
    K << SKEW_SYM_MATRX(r_axis);
    /// Roderigous Tranformation
    return I33 + std::sin(ang_norm) * K + (1.0 - std::cos(ang_norm)) * K * K;
  }
  
  return I33;
  
}

Eigen::Matrix3d Exp(const Eigen::Vector3d &ang_vel, const double &dt)
{
  double ang_vel_norm = ang_vel.norm();
  if (ang_vel_norm > 1e-7)
  {
    Eigen::Vector3d r_axis = ang_vel / ang_vel_norm;
    Eigen::Matrix3d K;

    K << SKEW_SYM_MATRX(r_axis);
    double r_ang = ang_vel_norm * dt;

    /// Roderigous Tranformation
    return I33 + std::sin(r_ang) * K + (1.0 - std::cos(r_ang)) * K * K;
  }
  
  return I33;
}

Eigen::Vector3d Log(const Eigen::Matrix3d &R)
{
  double theta = (R.trace() > 3.0 - 1e-6) ? 0.0 : std::acos(0.5 * (R.trace() - 1));
  Eigen::Vector3d K(R(2,1) - R(1,2), R(0,2) - R(2,0), R(1,0) - R(0,1));
  return (std::abs(theta) < 0.001) ? (0.5 * K) : (0.5 * theta / std::sin(theta) * K);
}

Eigen::Matrix3d hat(const Eigen::Vector3d &v)
{
  Eigen::Matrix3d Omega;
  Omega <<  0, -v(2),  v(1)
      ,  v(2),     0, -v(0)
      , -v(1),  v(0),     0;
  return Omega;
}

Eigen::Matrix3d jr(Eigen::Vector3d vec)
{
  double ang = vec.norm();

  if(ang < 1e-9)
  {
    return I33;
  }
  else
  {
    vec /= ang;
    double ra = sin(ang)/ang;
    return ra*I33 + (1-ra)*vec*vec.transpose() - (1-cos(ang))/ang * hat(vec);
  }
}

Eigen::Matrix3d jr_inv(const Eigen::Matrix3d &rotR)
{
  Eigen::AngleAxisd rot_vec(rotR);
  Eigen::Vector3d axi = rot_vec.axis();
  double ang = rot_vec.angle();

  if(ang < 1e-9)
  {
    return I33;
  }
  else
  {
    double ctt = ang / 2 / tan(ang/2);
    return ctt*I33 + (1-ctt)*axi*axi.transpose() + ang/2 * hat(axi);
  }
}

struct IMUST
{
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  double t;
  Eigen::Matrix3d R;
  Eigen::Vector3d p;
  Eigen::Vector3d v;
  Eigen::Vector3d bg;
  Eigen::Vector3d ba;
  Eigen::Vector3d g;
  Eigen::Matrix<double, DIM, DIM> cov;
  
  IMUST()
  {
    setZero();
  }

  IMUST(double _t, const Eigen::Matrix3d &_R, const Eigen::Vector3d &_p, const Eigen::Vector3d &_v, const Eigen::Vector3d &_bg, const Eigen::Vector3d &_ba, const Eigen::Vector3d &_g = Eigen::Vector3d(0, 0, -G_m_s2)) : t(_t), R(_R), p(_p), v(_v), bg(_bg), ba(_ba), g(_g) {}

  IMUST &operator+=(const Eigen::Matrix<double, DIM, 1> &ist)
  {
    this->R = this->R * Exp(ist.block<3, 1>(0, 0));
    this->p += ist.block<3, 1>(3, 0);
    this->v += ist.block<3, 1>(6, 0);
    this->bg += ist.block<3, 1>(9, 0);
    this->ba += ist.block<3, 1>(12, 0);
    return *this;
  }

  Eigen::Matrix<double, DIM, 1> operator-(const IMUST &b) 
  {
    Eigen::Matrix<double, DIM, 1> a;
    a.block<3, 1>(0, 0) = Log(b.R.transpose() * this->R);
    a.block<3, 1>(3, 0) = this->p - b.p;
    a.block<3, 1>(6, 0) = this->v - b.v;
    a.block<3, 1>(9, 0) = this->bg - b.bg;
    a.block<3, 1>(12, 0) = this->ba - b.ba;
    return a;
  }

  IMUST &operator=(const IMUST &b)
  {
    this->R = b.R;
    this->p = b.p;
    this->v = b.v;
    this->bg = b.bg;
    this->ba = b.ba;
    this->g = b.g;
    this->t = b.t;
    this->cov = b.cov;
    return *this;
  }

  void setZero()
  {
    t = 0; R.setIdentity();
    p.setZero(); v.setZero();
    bg.setZero(); ba.setZero();
    // g << 0, 0, -G_m_s2;
    cov.setIdentity(); 
    cov *= 0.0001;
    cov.block<6, 6>(9, 9) = Eigen::Matrix<double, 6, 6>::Identity() * 0.00001;
  }

};

// `average_xyz` false keeps the first real return in each cell instead of the
// running centroid of every return that fell in it. The centroid is not a
// measurement: with down_size 0.05 and ~24% survival a surviving point is the
// mean of ~4 returns spread over up to 5 cm, so it can sit a centimetre off the
// true surface, and a mixed pixel sharing a cell with good returns is partly
// averaged away before anything downstream can see it. Note that either way the
// non-xyz fields (intensity, ring, scan_time, ...) come from the first point in
// the cell, so with averaging on, a point's coordinates and its scalars describe
// different subsets of the data.
void down_sampling_voxel(pcl::PointCloud<PointType> &pl_feat, double voxel_size,
                         bool average_xyz = true)
{
  if(voxel_size < 0.001) return;

  unordered_map<VOXEL_LOC, PointType> feat_map;
  float loc_xyz[3];
  for(PointType &p_c : pl_feat.points)
  {
    for(int j=0; j<3; j++)
    {
      loc_xyz[j] = p_c.data[j] / voxel_size;
      if(loc_xyz[j] < 0)
        loc_xyz[j] -= 1.0;
    }

    VOXEL_LOC position((int64_t)loc_xyz[0], (int64_t)loc_xyz[1], (int64_t)loc_xyz[2]);
    auto iter = feat_map.find(position);
    if(iter == feat_map.end())
    {
      PointType pp = p_c;
      pp.curvature = 1;
      feat_map[position] = pp;
    }
    else
    {
      PointType &pp = iter->second;
      if(average_xyz)
      {
        pp.x = (pp.x * pp.curvature + p_c.x) / (pp.curvature + 1);
        pp.y = (pp.y * pp.curvature + p_c.y) / (pp.curvature + 1);
        pp.z = (pp.z * pp.curvature + p_c.z) / (pp.curvature + 1);
      }
      pp.curvature += 1;   // kept either way: it counts returns in the cell
    }
  }

  pl_feat.clear();
  for(auto iter=feat_map.begin(); iter!=feat_map.end(); ++iter)
    pl_feat.push_back(iter->second);
  
}
void down_sampling_voxel_xyz(pcl::PointCloud<pcl::PointXYZ> &pl_feat, double voxel_size)
{
  if(voxel_size < 0.001) return;

  unordered_map<VOXEL_LOC, pcl::PointXYZ> feat_map;
  float loc_xyz[3];
  for(pcl::PointXYZ &p_c : pl_feat.points)
  {
    for(int j=0; j<3; j++)
    {
      loc_xyz[j] = p_c.data[j] / voxel_size;
      if(loc_xyz[j] < 0)
        loc_xyz[j] -= 1.0;
    }

    VOXEL_LOC position((int64_t)loc_xyz[0], (int64_t)loc_xyz[1], (int64_t)loc_xyz[2]);
    auto iter = feat_map.find(position);
    if(iter == feat_map.end())
    {
      feat_map[position] = p_c;
    }
    else
    {
      pcl::PointXYZ &pp = iter->second;
      pp.x = (pp.x + p_c.x) / 2;
      pp.y = (pp.y + p_c.y) / 2;
      pp.z = (pp.z + p_c.z) / 2;
    }
  }

  pl_feat.clear();
  for(auto iter=feat_map.begin(); iter!=feat_map.end(); ++iter)
    pl_feat.push_back(iter->second);
}

void down_sampling_close(pcl::PointCloud<PointType> &pl_feat, double voxel_size)
{
  if(voxel_size < 0.001) return;

  unordered_map<VOXEL_LOC, pcl::PointCloud<PointType>::Ptr> feat_map;
  float loc_xyz[3];
  for(PointType &p_c: pl_feat.points)
  {
    for(int j=0; j<3; j++)
    {
      loc_xyz[j] = p_c.data[j] / voxel_size;
      if(loc_xyz[j] < 0)
        loc_xyz[j] -= 1.0;
    }

    VOXEL_LOC position((int64_t)loc_xyz[0], (int64_t)loc_xyz[1], (int64_t)loc_xyz[2]);
    auto iter = feat_map.find(position);
    if(iter == feat_map.end())
    {
      pcl::PointCloud<PointType>::Ptr pl_ptr(new pcl::PointCloud<PointType>);
      pl_ptr->push_back(p_c);
      feat_map[position] = pl_ptr;
    }
    else
    {
      iter->second->push_back(p_c);
    }
  }

  pl_feat.clear();
  for(auto iter=feat_map.begin(); iter!=feat_map.end(); ++iter)
  {
    pcl::PointCloud<PointType>::Ptr pl_ptr = iter->second;

    PointType pb = pl_ptr->points[0];
    int plsize = pl_ptr->size();
    for(int i=1; i<plsize; i++)
    {
      PointType &pp = pl_ptr->points[i];
      pb.x += pp.x; pb.y += pp.y; pb.z += pp.z;
    }
    pb.x /= plsize; pb.y /=plsize; pb.z /= plsize;

    double ndis = 100;
    int mnum = 0;
    for(int i=0; i<plsize; i++)
    {
      PointType &pp = pl_ptr->points[i];
      double xx = pb.x - pp.x;
      double yy = pb.y - pp.y;
      double zz = pb.z - pp.z;
      double dis = xx*xx + yy*yy + zz*zz;
      if(dis < ndis)
      {
        mnum = i;
        ndis = dis;
      }
    }

    pl_feat.push_back(pl_ptr->points[mnum]);
  }

}

class PointCluster
{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  Eigen::Matrix3d P;
  Eigen::Vector3d v;
  int N;

  PointCluster()
  {
    P.setZero();
    v.setZero();
    N = 0;
  }

  void clear()
  {
    P.setZero();
    v.setZero();
    N = 0;
  }

  void push(const Eigen::Vector3d &vec)
  {
    N++;
    P += vec * vec.transpose();
    v += vec;
  }

  Eigen::Matrix3d cov()
  {
    Eigen::Vector3d center = v / N;
    return P/N - center*center.transpose();
  }

  PointCluster & operator+=(const PointCluster &sigv)
  {
    this->P += sigv.P;
    this->v += sigv.v;
    this->N += sigv.N;

    return *this;
  }

  PointCluster & operator-=(const PointCluster &sigv)
  {
    this->P -= sigv.P;
    this->v -= sigv.v;
    this->N -= sigv.N;

    return *this;
  }

  void transform(const PointCluster &sigv, const IMUST &stat)
  {
    N = sigv.N;
    v = stat.R*sigv.v + N*stat.p;
    Eigen::Matrix3d rp = stat.R * sigv.v * stat.p.transpose();
    P = stat.R*sigv.P*stat.R.transpose() + rp + rp.transpose() + N*stat.p*stat.p.transpose();
  }

};

#endif