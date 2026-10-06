#ifndef FEATURE_POINT_HPP
#define FEATURE_POINT_HPP

#include <ros/ros.h>
#include <pcl_conversions/pcl_conversions.h>
#include <sensor_msgs/PointCloud2.h>
// #include <livox_ros_driver/CustomMsg.h>

// typedef pcl::PointXYZINormal PointType;
using namespace std;

enum LID_TYPE{LIVOX, VELODYNE, OUSTER, HESAI, ROBOSENSE, TARTANAIR, colored_ouster};

namespace velodyne_ros {
  struct EIGEN_ALIGN16 Point {
      PCL_ADD_POINT4D;
      // float intensity;
      float time;
      std::uint16_t ring;
      EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  };
}  // namespace velodyne_ros
POINT_CLOUD_REGISTER_POINT_STRUCT(velodyne_ros::Point,
    (float, x, x)
    (float, y, y)
    (float, z, z)
    // (float, intensity, intensity)
    (float, time, time)
    (std::uint16_t, ring, ring)
)

#include "range_image.hpp"

namespace ouster_ros 
{
  struct EIGEN_ALIGN16 Point 
  {
    PCL_ADD_POINT4D;
    float intensity;
    uint32_t t;
    uint16_t reflectivity;
    uint16_t ring;
    uint16_t ambient;
    uint32_t range;
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  };
}
POINT_CLOUD_REGISTER_POINT_STRUCT(ouster_ros::Point,
  (float, x, x)
  (float, y, y)
  (float, z, z)
  (float, intensity, intensity)
  // use std::uint32_t to avoid conflicting with pcl::uint32_t
  (std::uint32_t, t, t)
  (std::uint16_t, reflectivity, reflectivity)
  (std::uint16_t, ambient, ambient)
  (std::uint16_t, ring, ring)
  (std::uint32_t, range, range)
)

namespace xt32_ros {
  struct EIGEN_ALIGN16 Point {
      PCL_ADD_POINT4D;
      float intensity;
      double timestamp;
      uint16_t ring;
      EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  };
}  // namespace velodyne_ros
POINT_CLOUD_REGISTER_POINT_STRUCT(xt32_ros::Point,
    (float, x, x)
    (float, y, y)
    (float, z, z)
    (float, intensity, intensity)
    (double, timestamp, timestamp)
    (std::uint16_t, ring, ring)
)


namespace rslidar_ros {
  struct EIGEN_ALIGN16 Point {
      PCL_ADD_POINT4D;
      float intensity;
      std::uint16_t ring;
      double timestamp;
      EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  };
}
POINT_CLOUD_REGISTER_POINT_STRUCT(rslidar_ros::Point,
    (float, x, x)
    (float, y, y)
    (float, z, z)
    (float, intensity, intensity)
    (std::uint16_t, ring, ring)
    (double, timestamp, timestamp)
)

class Features
{
public:
  int lidar_type, point_filter_num;
  double blind = 1;
  double omega_l = 3610;

  // double process(const livox_ros_driver::CustomMsg::ConstPtr &msg, pcl::PointCloud<PointType> &pl_full)
  // {
  //   livox_handler(msg, pl_full);
  //   return msg->header.stamp.toSec();
  // }

  double process(const sensor_msgs::PointCloud2::ConstPtr &msg, pcl::PointCloud<PointType> &pl_full)
  {
    double t0 = msg->header.stamp.toSec();
    switch(lidar_type)
    {
    case VELODYNE:
      velodyne_handler(msg, pl_full);
      break;

    case OUSTER:
      ouster_handler(msg, pl_full);
      break;

    case HESAI:
      hesai_handler(msg, pl_full);
      break;
    
    case ROBOSENSE:
      t0 = robosense_handler(msg, pl_full);
      break;
    
    case TARTANAIR:
      tartanair_handler(msg, pl_full);
      break;

    case colored_ouster:
      {
        // Temporarily suppress PCL console output
        pcl::console::setVerbosityLevel(pcl::console::L_ALWAYS);
        // Direct conversion since /colored_points is already PointXYZIRGB
        pcl::fromROSMsg(*msg, pl_full); 
        // Restore normal verbosity
        pcl::console::setVerbosityLevel(pcl::console::L_INFO);
        // Apply filtering while preserving RGB
        pcl::PointCloud<PointType> filtered;
        for (int i = 0; i < pl_full.size(); i += point_filter_num) {
          PointType &pt = pl_full[i];
          if (pt.x*pt.x + pt.y*pt.y + pt.z*pt.z > blind) {
            pt.curvature = 0;  // No timing info needed for processed data
            filtered.push_back(pt);
          }
        } 
        pl_full = filtered;
        break;
      }
    default:
      printf("Lidar Type Error\n");
      exit(0);
    }

    return t0;
  }

  // void livox_handler(const livox_ros_driver::CustomMsg::ConstPtr &msg, pcl::PointCloud<PointType> &pl_full)
  // { 
  //   int plsize = msg->point_num;
  //   pl_full.reserve(plsize);

  //   for(int i=0; i<plsize; i++)
  //   {
  //     PointType ap;
  //     ap.x = msg->points[i].x;
  //     ap.y = msg->points[i].y;
  //     ap.z = msg->points[i].z;
  //     ap.intensity = msg->points[i].reflectivity;
  //     // ap.curvature = msg->points[i].offset_time / float(1000000); // ms
  //     ap.curvature = msg->points[i].offset_time / float(1000000000); // s

  //     if(i % point_filter_num == 0)
  //     {
  //       if(ap.x*ap.x + ap.y*ap.y + ap.z*ap.z > blind)
  //       {
  //         pl_full.push_back(ap);
  //       }
  //     }

  //   }

  // }

  void velodyne_handler(const sensor_msgs::PointCloud2::ConstPtr &msg, pcl::PointCloud<PointType> &pl_full)
  {
    pcl::PointCloud<velodyne_ros::Point> pl_orig;
    pcl::fromROSMsg(*msg, pl_orig);

    int plsize = pl_orig.size();
    if(plsize == 0) return;
    if(pl_orig.back().time > 0.01 && pl_orig.back().time < 0.12)
    {
      // for(velodyne_ros::Point &iter : pl_orig.points)
      for(int i=0; i<plsize; i++)
      {
        velodyne_ros::Point &iter = pl_orig[i];
        PointType ap;
        ap.x = iter.x; ap.y = iter.y; ap.z = iter.z;
        
        // ap.intensity = iter.intensity;
        // ap.curvature = iter.time * 1e-3; // ms
        // ap.curvature = iter.time * 1e-6;
        ap.curvature = iter.time;

        if(i % point_filter_num == 0)
        {
          if(ap.x*ap.x + ap.y*ap.y + ap.z*ap.z > blind)
          {
            pl_full.push_back(ap);
          }
        }
      }

    }
    else
    {
      // lidar clockwise rotate
      bool first_point = true;
      double yaw_first = 0;
      double yaw_last = 0;
      double yaw_bias = 0;
      int cool = 0;
      float max_ang = 0;
      for(int i=0; i<plsize; i++)
      {
        cool--;
        velodyne_ros::Point &iter = pl_orig[i];
        PointType ap;
        ap.x = iter.x; ap.y = iter.y; ap.z = iter.z;

        if(fabs(ap.x) < 0.1)
          continue;
        
        double yaw_angle = atan2(ap.y, ap.x) * 57.2957 - yaw_bias;
        if(first_point)
        {
          yaw_first = yaw_angle;
          yaw_last  = yaw_angle;
          first_point = false;
        }

        if(ap.x*ap.x + ap.y*ap.y + ap.z*ap.z < blind)
          continue;

        if(yaw_angle - yaw_last > 180 && cool <= 0)
        {
          yaw_bias += 360; yaw_angle-= 360; cool = 1000;
        }

        if(fabs(yaw_angle - yaw_last) > 180)
        {
          yaw_angle += 360;
        }

        ap.curvature = (yaw_first - yaw_angle) / omega_l;
        yaw_last = yaw_angle;

        if(ap.curvature > max_ang)
          max_ang = ap.curvature;

        if(ap.curvature >= 0 && ap.curvature < 0.1)
          if(i % point_filter_num == 0)
            pl_full.push_back(ap);
      }

      // printf("maxang: %f\n", max_ang);
    }

  }

  void ouster_handler(const sensor_msgs::PointCloud2::ConstPtr &msg, pcl::PointCloud<PointType> &pl_full)
  {
    pcl::PointCloud<ouster_ros::Point> pl_orig;
    pcl::fromROSMsg(*msg, pl_orig);
    
    int plsize = pl_orig.points.size();
    pl_full.reserve(plsize);

    // Within-scan detachment, computed here because this is the last point in
    // the pipeline where the scan is complete: the cloud is organised
    // (height=rings, width=azimuth columns) so a point's detector neighbours
    // are index arithmetic, and down_sampling_voxel has not yet removed ~76% of
    // the returns. See range_image.hpp for why the statistic is min/max of two
    // one-sided extrapolations rather than a neighbour difference.
    std::vector<float> ri_rng, ri_int, ri_det, ri_edg, ri_rgh,
                       ri_idet, ri_iedg;
    const bool organised = (pl_orig.height > 1 &&
                            int(pl_orig.height) * int(pl_orig.width) == plsize);
    if(organised)
    {
      ri_rng.resize(plsize); ri_int.resize(plsize);
      for(int i = 0; i < plsize; i++)
      {
        ri_rng[i] = float(pl_orig[i].range) * 1e-3f;   // driver reports mm
        ri_int[i] = pl_orig[i].intensity;
      }
      // columns wrap: the sweep is a full 360 deg, so col 0 and col width-1 are
      // azimuth neighbours.
      range_image::compute(ri_rng, ri_rng, pl_orig.height, pl_orig.width, true,
                           ri_det, ri_edg, &ri_rgh);
      range_image::compute(ri_int, ri_rng, pl_orig.height, pl_orig.width, true,
                           ri_idet, ri_iedg);
    }
    else
    {
      static bool warned = false;
      if(!warned)
      {
        printf("[ouster_handler] cloud is not organised (%ux%u); "
               "range-image features disabled\n", pl_orig.height, pl_orig.width);
        warned = true;
      }
    }

    for(int i = 0; i < plsize; i++)
    {
      PointType ap;
      ap.x = pl_orig.points[i].x;
      ap.y = pl_orig.points[i].y;
      ap.z = pl_orig.points[i].z;
      ap.intensity = pl_orig[i].intensity;
      ap.reflectivity = (float)pl_orig[i].reflectivity;
      ap.curvature = pl_orig[i].t / float(1e9); // s
      // Raw sensor channels the driver already gives us. curvature carries the
      // same `t` but is reused downstream as pv_var scratch, so scan_time keeps
      // its own slot. Only ouster_handler sets these; other lidar_type paths
      // leave them at whatever PointType was constructed with.
      ap.ambient      = (float)pl_orig[i].ambient;
      ap.ring         = (float)pl_orig[i].ring;
      ap.scan_time    = pl_orig[i].t / float(1e9); // s
      ap.range_detach = organised ? ri_det[i]  : 0.0f;
      ap.range_edge   = organised ? ri_edg[i]  : 0.0f;
      ap.int_detach   = organised ? ri_idet[i] : 0.0f;
      ap.ring_rough   = organised ? ri_rgh[i]  : 0.0f;
      
      // Initialize RGB fields (gray for non-colored points)
      ap.r = 128;
      ap.g = 128;
      ap.b = 128;
      ap.a = 255;
      ap.rgb = ((std::uint32_t)128 << 16) | ((std::uint32_t)128 << 8) | 128;
      
      // Initialize normal fields
      ap.normal_x = 0;
      ap.normal_y = 0;
      ap.normal_z = 0;
      
      if(i % point_filter_num == 0)
      {
        if(ap.x*ap.x + ap.y*ap.y + ap.z*ap.z > blind)
        {
          pl_full.points.push_back(ap);
        }
      }
    }
  }

  // Hesai path. Measured on the Oxford Spires QT64 (Keble college-02):
  //   * the message is height=1, width~60016 -- a flat list, NOT an organised
  //     range image like the Ouster driver produces;
  //   * but the sampling underneath is perfectly regular: 64 rings x 600
  //     columns at exactly 0.600 deg azimuth (100% of gaps are integer
  //     multiples of it), so a range image IS reconstructable from ring +
  //     atan2(y,x) should family B be wanted here;
  //   * the sensor runs in DUAL RETURN mode: 99.8% of (ring, column) cells hold
  //     exactly two points, and in 98.3% of those the two ranges are identical.
  //     So roughly half of every scan is an exact duplicate. Deduplicating to
  //     one return per cell would halve the data at almost no information cost.
  //
  // The grid IS now rebuilt, so family B is available on this sensor; see
  // range_image::compute_scattered.
  void hesai_handler(const sensor_msgs::PointCloud2::ConstPtr &msg, pcl::PointCloud<PointType> &pl_full)
  { 
    pcl::PointCloud<xt32_ros::Point> pl_orig;
    pcl::fromROSMsg(*msg, pl_orig);

    int plsize = pl_orig.points.size();
    pl_full.reserve(plsize);
    if(plsize == 0) return;
    double time_head = pl_orig.points[0].timestamp;

    // Within-scan detachment. The driver publishes a flat list, so unlike the
    // Ouster path the (ring, azimuth column) lattice has to be rebuilt first --
    // see range_image::compute_scattered. Done here for the same reason as on
    // the Ouster: this is the last place the scan is complete, before
    // down_sampling_voxel removes most of the returns.
    std::vector<float> ri_det, ri_edg, ri_rgh, ri_idet, ri_iedg;
    bool ri_ok = false;
    {
      std::vector<float> rng(plsize), inten(plsize), az(plsize);
      std::vector<int> row(plsize), col(plsize);
      int max_ring = 0;
      for(int i = 0; i < plsize; i++)
      {
        const auto &p = pl_orig.points[i];
        rng[i]   = std::sqrt(p.x * p.x + p.y * p.y + p.z * p.z);
        inten[i] = p.intensity;
        float a  = std::atan2(p.y, p.x) * 57.29577951308232f;   // deg
        if(a < 0.0f) a += 360.0f;
        az[i]  = a;
        row[i] = int(p.ring);
        if(row[i] > max_ring) max_ring = row[i];
      }
      const int height = max_ring + 1;
      // Estimated once per process: the geometry is a property of the sensor
      // and its rate, not of the individual scan, and re-estimating every scan
      // would risk the grid changing under a run.
      static int ncol = -1;
      if(ncol < 0)
      {
        ncol = range_image::estimate_columns(az, row, max_ring / 2);
        if(ncol > 0)
          printf("[hesai_handler] azimuth grid: %d rings x %d columns "
                 "(%.4f deg); range-image features enabled\n",
                 height, ncol, 360.0 / ncol);
        else
          printf("[hesai_handler] could not establish an azimuth grid; "
                 "range-image features disabled\n");
      }
      if(ncol > 0)
      {
        const float step = 360.0f / float(ncol);
        for(int i = 0; i < plsize; i++)
        {
          int c = int(std::lround(az[i] / step));
          col[i] = ((c % ncol) + ncol) % ncol;      // 360 deg wraps to column 0
        }
        range_image::compute_scattered(rng, rng, row, col, height, ncol, true,
                                       ri_det, ri_edg, &ri_rgh);
        range_image::compute_scattered(inten, rng, row, col, height, ncol, true,
                                       ri_idet, ri_iedg);
        ri_ok = true;
      }
    }

    for(int i=0; i<plsize; i++)
    {
      PointType added_pt;

      added_pt.normal_x = 0;
      added_pt.normal_y = 0;
      added_pt.normal_z = 0;
      added_pt.x = pl_orig.points[i].x;
      added_pt.y = pl_orig.points[i].y;
      added_pt.z = pl_orig.points[i].z;
      added_pt.intensity = pl_orig.points[i].intensity;
      added_pt.curvature = (pl_orig.points[i].timestamp - time_head);

      // Every channel PointType carries must be written here. PointType is a
      // plain struct with no constructor, so `PointType added_pt;` leaves each
      // field indeterminate -- it holds whatever was on the stack. Anything not
      // assigned would therefore be saved as a plausible-looking float rather
      // than an obvious "missing", which is far worse than a zero.
      //
      // Available on the Hesai and now carried through:
      added_pt.ring      = (float)pl_orig.points[i].ring;
      added_pt.scan_time = (float)(pl_orig.points[i].timestamp - time_head); // s
      // Not measured by this sensor -- zero means "channel absent", and the
      // dataset config switches these off in SaveFields anyway:
      added_pt.reflectivity = 0.0f;   // Ouster-only calibrated reflectivity
      added_pt.ambient      = 0.0f;   // Ouster-only background NIR
      // Family B, from the rebuilt azimuth grid above. 0 where no grid could be
      // established, or where a point has no complete window on both sides --
      // the same "no evidence of detachment" default the Ouster path uses.
      added_pt.range_detach = ri_ok ? ri_det[i]  : 0.0f;
      added_pt.range_edge   = ri_ok ? ri_edg[i]  : 0.0f;
      added_pt.int_detach   = ri_ok ? ri_idet[i] : 0.0f;
      added_pt.ring_rough   = ri_ok ? ri_rgh[i]  : 0.0f;
      // Gray, as ouster_handler does for non-colored points.
      added_pt.r = 128; added_pt.g = 128; added_pt.b = 128; added_pt.a = 255;
      added_pt.rgb = ((std::uint32_t)128 << 16) | ((std::uint32_t)128 << 8) | 128;

      if (i % point_filter_num == 0)
      {
        if (added_pt.x*added_pt.x+added_pt.y*added_pt.y+added_pt.z*added_pt.z > blind)
        {
          pl_full.points.push_back(added_pt);
        }
      }


    }

  }

  double robosense_handler(const sensor_msgs::PointCloud2::ConstPtr &msg, pcl::PointCloud<PointType> &pl_full)
  {
    pcl::PointCloud<rslidar_ros::Point> pl_orig;
    pcl::fromROSMsg(*msg, pl_orig);

    int plsize = pl_orig.points.size();
    pl_full.reserve(plsize);
    double t0 = pl_orig[0].timestamp;
    for(int i=0; i<plsize; i++)
    {
      PointType ap;
      ap.x = pl_orig.points[i].x;
      ap.y = pl_orig.points[i].y;
      ap.z = pl_orig.points[i].z;
      ap.intensity = pl_orig.points[i].intensity;
      // ap.curvature = (pl_orig[i].timestamp - t0) * float(1e3); //
      ap.curvature = (pl_orig[i].timestamp - t0);

      if(i % point_filter_num == 0)
      {
        if(ap.x*ap.x + ap.y*ap.y + ap.z*ap.z > blind)
        {
          pl_full.points.push_back(ap);
        }
      }

    }

    return t0;
  }

  void tartanair_handler(const sensor_msgs::PointCloud2::ConstPtr &msg, pcl::PointCloud<PointType> &pl_full)
  {
    pcl::PointCloud<pcl::PointXYZ> pl_orig;
    pcl::fromROSMsg(*msg, pl_orig);
    pl_full.reserve(pl_orig.size());

    PointType pp; pp.curvature = 0;
    for(pcl::PointXYZ &ap: pl_orig.points)
    {
      pp.x = ap.x;
      pp.y = ap.y;
      pp.z = ap.z; 
      pl_full.push_back(pp);
    }

    return;
  }

};

#endif
