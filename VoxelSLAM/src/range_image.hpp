#ifndef RANGE_IMAGE_HPP
#define RANGE_IMAGE_HPP

/* Within-scan detachment features, computed on the raw organised scan.
 *
 * Every other confidence channel in this project describes a 3D neighbourhood
 * on the *merged* cloud. That is the wrong neighbourhood for mixed pixels: a
 * mixed pixel is created when one beam straddles a depth discontinuity and the
 * returned range is a blend of foreground and background, so the evidence lives
 * between a point and its neighbours *on the detector* -- same ring, adjacent
 * azimuth. Once ~1900 scans are merged, other viewpoints have filled the space
 * around that point and the evidence is gone. No per-row model can recover it
 * either: computing the range difference to the point at (ring 37, column 431)
 * is a join across rows, which a per-point regressor cannot express.
 *
 * Computed here, in the handler, because this is the last place the data is
 * complete: down_sampling_voxel later keeps ~24% of returns, and because a
 * flyer alone in empty space is the only occupant of its voxel it always
 * survives while the dense surface around it is thinned -- so the feature would
 * be measuring a cloud stripped of exactly the context it needs.
 *
 * Statistic, per side, is a one-sided linear extrapolation:
 *
 *     slope    = median per-column increment over K samples on one side
 *     pred     = nearest used sample + slope * (1 + SKIP)
 *     residual = |range - pred|
 *
 *     detach = min(residual_left, residual_right)   "am I on a surface?"
 *     edge   = max(residual_left, residual_right)   "how big is the cliff?"
 *
 * Why one-sided, then min:
 *   - a smooth surface, however steep, is extrapolated correctly from both
 *     sides. A grazing wall at 10 m can step 35 cm per column and still be
 *     predicted exactly; a raw neighbour difference would flag it.
 *   - at a clean depth edge the point is flush with its own surface, so the
 *     side it belongs to predicts it and min() takes that side.
 *   - a mixed pixel belongs to neither surface: both residuals large.
 * Fitting a line *through* the point using both sides at once does not work --
 * at an edge it fits across the discontinuity and every nearby point gets a
 * large residual (measured: 39% of points flagged).
 *
 * SKIP excludes the immediately adjacent columns. Mixed pixels come in runs of
 * two or three, and without it a neighbouring mixed pixel anchors the
 * extrapolation onto itself, the residual collapses to ~0, min() takes that
 * side and the whole run is missed. Measured on a synthetic pair, SKIP=0 scored
 * them *below* a clean edge.
 *
 * Both outputs are divided by the point's own range. Un-normalised they inherit
 * a strong distance bias -- the top 5% by absolute detachment sat 2.8 m beyond
 * the scene mean range -- because beam footprint and per-column range step both
 * grow with distance. That is the "distance proxy in disguise" failure that
 * made pv_var and pose_unc_rot useless.
 */

#include <vector>
#include <cmath>
#include <algorithm>
#include <cstdint>

namespace range_image
{

// K=3 rather than a larger window, which is counter-intuitive and was measured.
// For the first point of a run of mixed pixels the far side's anchor nb[0] is
// itself a flyer. With K>=4 the increments look like [-2, 0, 0] and the MEDIAN
// robustly returns 0, so the prediction stays on the flyer and the residual
// collapses to ~0 -- min() then takes that side and the run is missed. K=3
// leaves two increments whose average is dragged by the contaminated one, which
// pushes the prediction away and catches it. Synthetic runs of four scored
// 0.10% at K=4 (below a clean edge, i.e. invisible) against 28.6% at K=3; on a
// real scan K=3 gives contrast p99/p50 of 145 against 89, flags 2.4x as many
// points, and has identical false-positive behaviour on ramps and flats.
static const int   RI_K       = 3;      // samples per side
static const int   RI_SKIP    = 1;      // adjacent columns excluded
static const float RI_MIN_RNG = 0.10f;  // below this a return is not usable

// Median of a tiny fixed-size buffer; n <= RI_K, so a sort is cheaper than
// anything cleverer.
inline float small_median(float *v, int n)
{
  std::sort(v, v + n);
  return (n & 1) ? v[n / 2] : 0.5f * (v[n / 2 - 1] + v[n / 2]);
}

// One side of one row. `get(c)` returns the range at column c, or <=0 if the
// cell has no usable return. `dir` is -1 for the lower-column side, +1 for the
// higher. Returns false when the window is incomplete.
template <typename Getter>
inline bool one_side(const Getter &get, int col, int width, int dir,
                     bool wrap, float &residual, float &spread, float self)
{
  float nb[RI_K];
  for(int j = 0; j < RI_K; j++)
  {
    int c = col + dir * (j + 1 + RI_SKIP);
    if(wrap) c = (c % width + width) % width;
    else if(c < 0 || c >= width) return false;
    float r = get(c);
    if(!(r > RI_MIN_RNG)) return false;     // a gap makes the window unusable
    nb[j] = r;
  }
  float slopes[RI_K - 1];
  for(int j = 0; j < RI_K - 1; j++)
    slopes[j] = nb[j] - nb[j + 1];
  // The spread of the increments is free here and measures something the
  // residual does not: how consistent the surface is along the ring. A grazing
  // wall has large but *equal* increments, so its spread is ~0 -- which is the
  // whole point, it is the same normalisation that keeps `detach` from firing
  // on grazing surfaces. Rough or broken structure has scattered increments.
  float smin = slopes[0], smax = slopes[0];
  for(int j = 1; j < RI_K - 1; j++)
  {
    if(slopes[j] < smin) smin = slopes[j];
    if(slopes[j] > smax) smax = slopes[j];
  }
  spread = smax - smin;
  float slope = small_median(slopes, RI_K - 1);
  float pred  = nb[0] + slope * float(1 + RI_SKIP);
  residual = std::fabs(self - pred);
  return true;
}

/* Fill `detach` and `edge` (both normalised by the point's own value) for one
 * organised scan. `value` is what the detachment is measured on: range for the
 * geometric version, intensity for the photometric one. `range` gates validity
 * either way, because a cell with no return has no meaningful intensity.
 *
 * Points without a full window on both sides get 0, which reads as "no evidence
 * of detachment" -- the conservative default, and it avoids a sentinel
 * colliding with legitimate small values.
 */
inline void compute(const std::vector<float> &value,
                    const std::vector<float> &range,
                    int height, int width, bool wrap,
                    std::vector<float> &detach, std::vector<float> &edge,
                    std::vector<float> *rough = nullptr)
{
  const size_t n = size_t(height) * size_t(width);
  detach.assign(n, 0.0f);
  edge.assign(n, 0.0f);
  if(rough) rough->assign(n, 0.0f);

  for(int row = 0; row < height; row++)
  {
    const size_t base = size_t(row) * size_t(width);
    auto get = [&](int c) -> float {
      return (range[base + size_t(c)] > RI_MIN_RNG) ? value[base + size_t(c)]
                                                    : -1.0f;
    };
    for(int col = 0; col < width; col++)
    {
      const size_t i = base + size_t(col);
      if(!(range[i] > RI_MIN_RNG)) continue;
      const float self = value[i];
      float rl, rr, sl, sr;
      if(!one_side(get, col, width, -1, wrap, rl, sl, self)) continue;
      if(!one_side(get, col, width, +1, wrap, rr, sr, self)) continue;
      const float scale = std::max(std::fabs(self), 1e-3f);
      detach[i] = std::min(rl, rr) / scale;
      edge[i]   = std::max(rl, rr) / scale;
      // the calmer of the two sides: a point next to one edge still has one
      // well-behaved side, and that is the side its detachment was judged on
      if(rough) (*rough)[i] = std::min(sl, sr) / scale;
    }
  }
}

/* Same statistic, for a sensor whose driver publishes a flat list even though
 * the sampling underneath is a regular (row, column) lattice.
 *
 * The Ouster driver hands over an organised cloud, so a point's detector
 * neighbours are index arithmetic and compute() can be called directly. Others
 * do not. The Hesai QT64 in the Oxford Spires dataset publishes height=1,
 * width~60016 -- but measurement shows the returns sit on an exact 64 x 600
 * grid at 0.600 deg azimuth (100% of within-ring azimuth gaps are integer
 * multiples of that step), so the grid can simply be rebuilt.
 *
 * Two differences from an organised cloud, both handled here:
 *
 *   - CELLS CAN BE EMPTY. A beam with no return contributes no point at all,
 *     rather than a zero-range entry, so occupancy is ~78% not ~100%. Empty
 *     cells keep range 0, which one_side() already treats as a gap that makes
 *     the window unusable -- the same behaviour as a no-return on the Ouster.
 *
 *   - CELLS CAN HOLD SEVERAL POINTS. In dual-return mode 99.8% of cells hold
 *     exactly two, and in 98.3% of those the two ranges are identical. The
 *     nearest return per cell defines the grid, and the computed value is then
 *     scattered back to EVERY point of that cell, so both returns of a pair
 *     carry the feature.
 */
inline void compute_scattered(const std::vector<float> &value,
                              const std::vector<float> &range,
                              const std::vector<int> &row,
                              const std::vector<int> &col,
                              int height, int width, bool wrap,
                              std::vector<float> &detach,
                              std::vector<float> &edge,
                              std::vector<float> *rough = nullptr)
{
  const size_t np = value.size();
  const size_t ng = size_t(height) * size_t(width);

  std::vector<float> g_val(ng, 0.0f), g_rng(ng, 0.0f);
  for(size_t i = 0; i < np; i++)
  {
    if(!(range[i] > RI_MIN_RNG)) continue;
    if(row[i] < 0 || row[i] >= height || col[i] < 0 || col[i] >= width) continue;
    const size_t k = size_t(row[i]) * size_t(width) + size_t(col[i]);
    // nearest return wins the cell; ties keep the first seen
    if(g_rng[k] <= RI_MIN_RNG || range[i] < g_rng[k])
    {
      g_rng[k] = range[i];
      g_val[k] = value[i];
    }
  }

  std::vector<float> g_det, g_edg, g_rgh;
  compute(g_val, g_rng, height, width, wrap, g_det, g_edg,
          rough ? &g_rgh : nullptr);

  detach.assign(np, 0.0f);
  edge.assign(np, 0.0f);
  if(rough) rough->assign(np, 0.0f);
  for(size_t i = 0; i < np; i++)
  {
    if(row[i] < 0 || row[i] >= height || col[i] < 0 || col[i] >= width) continue;
    const size_t k = size_t(row[i]) * size_t(width) + size_t(col[i]);
    detach[i] = g_det[k];
    edge[i]   = g_edg[k];
    if(rough) (*rough)[i] = g_rgh[k];
  }
}

/* Number of azimuth columns in a sweep, estimated from the data.
 *
 * Hard-coding 600 would be wrong the moment the sensor is run at a different
 * rotation rate or azimuth resolution, and the failure would be silent: the
 * grid would alias and the features would quietly become noise. The step is
 * instead read off the scan as the median positive azimuth gap within one
 * well-populated ring, which is robust to the gaps left by non-returns because
 * those show up as integer multiples of the base step.
 *
 * Returns 0 if no reliable estimate can be made, and the caller should then
 * skip the features rather than guess.
 */
inline int estimate_columns(const std::vector<float> &azimuth_deg,
                            const std::vector<int> &row, int probe_row)
{
  std::vector<float> a;
  a.reserve(1024);
  for(size_t i = 0; i < azimuth_deg.size(); i++)
    if(row[i] == probe_row) a.push_back(azimuth_deg[i]);
  if(a.size() < 64) return 0;
  std::sort(a.begin(), a.end());
  std::vector<float> gaps;
  gaps.reserve(a.size());
  for(size_t i = 1; i < a.size(); i++)
  {
    float d = a[i] - a[i - 1];
    if(d > 1e-4f) gaps.push_back(d);
  }
  if(gaps.size() < 32) return 0;
  std::nth_element(gaps.begin(), gaps.begin() + gaps.size() / 2, gaps.end());
  const float step = gaps[gaps.size() / 2];
  if(!(step > 1e-3f) || step > 5.0f) return 0;
  const int ncol = int(std::lround(360.0f / step));
  return (ncol >= 180 && ncol <= 7200) ? ncol : 0;
}

}  // namespace range_image
#endif
