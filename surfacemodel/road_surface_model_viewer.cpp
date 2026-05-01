#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <GL/glu.h>
#include <GL/freeglut.h>

#include <Eigen/Dense>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <random>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

// ------------------------------------------------------------
// Tunable settings
// ------------------------------------------------------------
static const double VOXEL_SIZE_MM = 20.0;       // downsample size
static const double PLANE_THRESH_MM = 25.0;     // RANSAC plane threshold
static const int    RANSAC_ITERS = 1200;

static const double GRID_RES_MM = 30.0;         // surface model grid resolution
static const int    SMOOTH_ITERS = 7;           // surface smoothing passes
static const double HEIGHT_EXAGGERATION = 3.0;  // makes small road changes visible

// Percentile crop to remove extreme points and near-camera car hood
static const double X_KEEP_LOW = 3.0;
static const double X_KEEP_HIGH = 97.0;
static const double Y_KEEP_LOW = 3.0;
static const double Y_KEEP_HIGH = 97.0;
static const double Z_KEEP_LOW = 15.0;          // remove nearest points, often hood
static const double Z_KEEP_HIGH = 97.0;

// Remove bottom portion of point cloud to avoid vehicle hood.
// Axis: 0 = x, 1 = y, 2 = z.
// For image-bottom style cropping, start with y-axis.
static const int    BOTTOM_CROP_AXIS = 1;
static const double BOTTOM_REMOVE_PCT = 20.0;

// If hood is still visible, flip this between true/false.
static const bool   REMOVE_LOW_END = false;

// ------------------------------------------------------------
// Data structures
// ------------------------------------------------------------
struct Point3D {
    double x, y, z;
    unsigned char r = 180, g = 180, b = 180;
};

struct SurfaceGrid {
    int nx = 0;
    int ny = 0;
    double x_min = 0;
    double y_min = 0;
    double x_center = 0;
    double y_center = 0;
    double res_mm = GRID_RES_MM;
    std::vector<double> h;       // height in mm
    std::vector<unsigned char> valid;
    double h_min = 0;
    double h_max = 0;
};

struct VoxelKey {
    int x, y, z;

    bool operator==(const VoxelKey& other) const {
        return x == other.x && y == other.y && z == other.z;
    }
};

struct VoxelHash {
    std::size_t operator()(const VoxelKey& k) const {
        std::size_t h1 = std::hash<int>()(k.x);
        std::size_t h2 = std::hash<int>()(k.y);
        std::size_t h3 = std::hash<int>()(k.z);
        return h1 ^ (h2 << 1) ^ (h3 << 2);
    }
};

struct Accum {
    double x = 0, y = 0, z = 0;
    double r = 0, g = 0, b = 0;
    int count = 0;
};

// ------------------------------------------------------------
// Utility functions
// ------------------------------------------------------------
double percentile(std::vector<double> values, double pct) {
    if (values.empty()) return 0.0;

    pct = std::max(0.0, std::min(100.0, pct));
    size_t idx = static_cast<size_t>((pct / 100.0) * (values.size() - 1));

    std::nth_element(values.begin(), values.begin() + idx, values.end());
    return values[idx];
}

std::vector<double> collectAxis(const std::vector<Point3D>& pts, int axis) {
    std::vector<double> vals;
    vals.reserve(pts.size());

    for (const auto& p : pts) {
        if (axis == 0) vals.push_back(p.x);
        else if (axis == 1) vals.push_back(p.y);
        else vals.push_back(p.z);
    }

    return vals;
}

Eigen::Vector3d toEigen(const Point3D& p) {
    return Eigen::Vector3d(p.x, p.y, p.z);
}

static std::string trimLine(std::string s) {
    while (!s.empty() &&
           (s.back() == '\r' || s.back() == '\n' ||
            s.back() == ' '  || s.back() == '\t')) {
        s.pop_back();
    }

    size_t start = 0;
    while (start < s.size() &&
           (s[start] == ' ' || s[start] == '\t')) {
        start++;
    }

    return s.substr(start);
}

double getAxisValue(const Point3D& p, int axis) {
    if (axis == 0) return p.x;
    if (axis == 1) return p.y;
    return p.z;
}

// ------------------------------------------------------------
// Robust ASCII PLY reader
// Works even if xyz and rgb appear split across lines.
// ------------------------------------------------------------
bool readAsciiPly(const std::string& path, std::vector<Point3D>& points) {
    std::ifstream file(path);
    if (!file.is_open()) {
        std::cerr << "Failed to open PLY: " << path << std::endl;
        return false;
    }

    std::string line;
    size_t vertex_count = 0;
    bool header_done = false;

    while (std::getline(file, line)) {
        line = trimLine(line);


        if (line.rfind("element vertex", 0) == 0) {
            std::stringstream ss(line);
            std::string tmp1, tmp2;
            ss >> tmp1 >> tmp2 >> vertex_count;
        }

        if (line == "end_header") {
            header_done = true;
            break;
        }
    }

    if (!header_done || vertex_count == 0) {
        std::cerr << "Invalid PLY header or missing vertex count." << std::endl;
        return false;
    }

    std::cout << "PLY vertices listed: " << vertex_count << std::endl;

    std::vector<double> vals;
    vals.reserve(vertex_count * 6);

    double v;
    while (file >> v && vals.size() < vertex_count * 6) {
        vals.push_back(v);
    }

    if (vals.size() < vertex_count * 6) {
        std::cerr << "Warning: expected " << vertex_count * 6
                  << " vertex values, got " << vals.size() << std::endl;
    }

    size_t n = vals.size() / 6;
    points.reserve(n);

    for (size_t i = 0; i < n; ++i) {
        Point3D p;
        p.x = vals[i * 6 + 0];
        p.y = vals[i * 6 + 1];
        p.z = vals[i * 6 + 2];

        p.r = static_cast<unsigned char>(std::clamp(vals[i * 6 + 3], 0.0, 255.0));
        p.g = static_cast<unsigned char>(std::clamp(vals[i * 6 + 4], 0.0, 255.0));
        p.b = static_cast<unsigned char>(std::clamp(vals[i * 6 + 5], 0.0, 255.0));

        points.push_back(p);
    }

    std::cout << "Loaded points: " << points.size() << std::endl;
    return true;
}

// ------------------------------------------------------------
// Voxel downsample
// ------------------------------------------------------------
std::vector<Point3D> voxelDownsample(const std::vector<Point3D>& pts, double voxel_mm) {
    std::unordered_map<VoxelKey, Accum, VoxelHash> voxels;

    for (const auto& p : pts) {
        VoxelKey key {
            static_cast<int>(std::floor(p.x / voxel_mm)),
            static_cast<int>(std::floor(p.y / voxel_mm)),
            static_cast<int>(std::floor(p.z / voxel_mm))
        };

        auto& a = voxels[key];
        a.x += p.x;
        a.y += p.y;
        a.z += p.z;
        a.r += p.r;
        a.g += p.g;
        a.b += p.b;
        a.count++;
    }

    std::vector<Point3D> out;
    out.reserve(voxels.size());

    for (const auto& kv : voxels) {
        const auto& a = kv.second;
        Point3D p;
        p.x = a.x / a.count;
        p.y = a.y / a.count;
        p.z = a.z / a.count;
        p.r = static_cast<unsigned char>(a.r / a.count);
        p.g = static_cast<unsigned char>(a.g / a.count);
        p.b = static_cast<unsigned char>(a.b / a.count);
        out.push_back(p);
    }

    std::cout << "After voxel downsample: " << out.size() << " points\n";
    return out;
}

// ------------------------------------------------------------
// Percentile crop
// ------------------------------------------------------------

std::vector<Point3D> removeBottomPercentOfCloud(const std::vector<Point3D>& pts) {
    std::vector<double> values;
    values.reserve(pts.size());

    for (const auto& p : pts) {
        values.push_back(getAxisValue(p, BOTTOM_CROP_AXIS));
    }

    double cut;

    if (REMOVE_LOW_END) {
        cut = percentile(values, BOTTOM_REMOVE_PCT);
    } else {
        cut = percentile(values, 100.0 - BOTTOM_REMOVE_PCT);
    }

    std::vector<Point3D> out;
    out.reserve(pts.size());

    for (const auto& p : pts) {
        double v = getAxisValue(p, BOTTOM_CROP_AXIS);

        if (REMOVE_LOW_END) {
            // Remove lowest 20%
            if (v >= cut) {
                out.push_back(p);
            }
        } else {
            // Remove highest 20%
            if (v <= cut) {
                out.push_back(p);
            }
        }
    }

    std::cout << "After bottom " << BOTTOM_REMOVE_PCT
              << "% cloud crop: " << out.size() << " points\n";

    return out;
}


std::vector<Point3D> cropPercentile(const std::vector<Point3D>& pts) {
    auto xs = collectAxis(pts, 0);
    auto ys = collectAxis(pts, 1);
    auto zs = collectAxis(pts, 2);

    double x_lo = percentile(xs, X_KEEP_LOW);
    double x_hi = percentile(xs, X_KEEP_HIGH);
    double y_lo = percentile(ys, Y_KEEP_LOW);
    double y_hi = percentile(ys, Y_KEEP_HIGH);
    double z_lo = percentile(zs, Z_KEEP_LOW);
    double z_hi = percentile(zs, Z_KEEP_HIGH);

    std::vector<Point3D> out;
    out.reserve(pts.size());

    for (const auto& p : pts) {
        if (p.x >= x_lo && p.x <= x_hi &&
            p.y >= y_lo && p.y <= y_hi &&
            p.z >= z_lo && p.z <= z_hi) {
            out.push_back(p);
        }
    }

    std::cout << "After crop: " << out.size() << " points\n";
    return out;
}

// ------------------------------------------------------------
// RANSAC plane fitting
// Plane: normal dot x + d = 0
// ------------------------------------------------------------
bool fitPlaneRANSAC(
    const std::vector<Point3D>& pts,
    Eigen::Vector3d& best_normal,
    double& best_d,
    Eigen::Vector3d& plane_center
) {
    if (pts.size() < 100) {
        std::cerr << "Not enough points for plane fitting.\n";
        return false;
    }

    std::mt19937 rng(42);
    std::uniform_int_distribution<size_t> dist(0, pts.size() - 1);

    int best_inliers = -1;
    Eigen::Vector3d temp_best_normal(0, 0, 1);
    double temp_best_d = 0;

    for (int iter = 0; iter < RANSAC_ITERS; ++iter) {
        size_t i0 = dist(rng);
        size_t i1 = dist(rng);
        size_t i2 = dist(rng);

        if (i0 == i1 || i1 == i2 || i0 == i2) continue;

        Eigen::Vector3d p0 = toEigen(pts[i0]);
        Eigen::Vector3d p1 = toEigen(pts[i1]);
        Eigen::Vector3d p2 = toEigen(pts[i2]);

        Eigen::Vector3d n = (p1 - p0).cross(p2 - p0);
        double norm = n.norm();

        if (norm < 1e-9) continue;

        n.normalize();
        double d = -n.dot(p0);

        int inliers = 0;
        for (const auto& p : pts) {
            double distance = std::abs(n.dot(toEigen(p)) + d);
            if (distance < PLANE_THRESH_MM) {
                inliers++;
            }
        }

        if (inliers > best_inliers) {
            best_inliers = inliers;
            temp_best_normal = n;
            temp_best_d = d;
        }
    }

    std::cout << "RANSAC plane inliers: " << best_inliers
              << " / " << pts.size() << std::endl;

    if (best_inliers < 100) {
        std::cerr << "Plane fit failed.\n";
        return false;
    }

    // Refine plane with inliers using covariance/SVD
    std::vector<Eigen::Vector3d> inlier_pts;
    inlier_pts.reserve(best_inliers);

    for (const auto& p : pts) {
        Eigen::Vector3d v = toEigen(p);
        double distance = std::abs(temp_best_normal.dot(v) + temp_best_d);
        if (distance < PLANE_THRESH_MM) {
            inlier_pts.push_back(v);
        }
    }

    Eigen::Vector3d centroid(0, 0, 0);
    for (const auto& p : inlier_pts) centroid += p;
    centroid /= static_cast<double>(inlier_pts.size());

    Eigen::Matrix3d cov = Eigen::Matrix3d::Zero();
    for (const auto& p : inlier_pts) {
        Eigen::Vector3d q = p - centroid;
        cov += q * q.transpose();
    }

    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solver(cov);
    Eigen::Vector3d normal = solver.eigenvectors().col(0);
    normal.normalize();

    double d = -normal.dot(centroid);

    best_normal = normal;
    best_d = d;
    plane_center = centroid;

    std::cout << "Refined plane: "
              << best_normal.x() << "x + "
              << best_normal.y() << "y + "
              << best_normal.z() << "z + "
              << best_d << " = 0\n";

    return true;
}

// ------------------------------------------------------------
// Level point cloud: rotate plane normal to Z axis
// ------------------------------------------------------------
std::vector<Point3D> levelPoints(
    const std::vector<Point3D>& pts,
    Eigen::Vector3d normal,
    const Eigen::Vector3d& plane_center
) {
    Eigen::Vector3d z_axis(0, 0, 1);

    if (normal.z() < 0) {
        normal = -normal;
    }

    Eigen::Quaterniond q = Eigen::Quaterniond::FromTwoVectors(normal, z_axis);
    Eigen::Matrix3d R = q.toRotationMatrix();

    std::vector<Point3D> leveled;
    leveled.reserve(pts.size());

    std::vector<double> z_values;
    z_values.reserve(pts.size());

    for (const auto& p : pts) {
        Eigen::Vector3d v = toEigen(p);
        Eigen::Vector3d lv = R * (v - plane_center);

        Point3D out = p;
        out.x = lv.x();
        out.y = lv.y();
        out.z = lv.z();

        leveled.push_back(out);
        z_values.push_back(out.z);
    }

    // Put fitted road plane near z = 0 using median height.
    double z_med = percentile(z_values, 50.0);

    for (auto& p : leveled) {
        p.z -= z_med;
    }

    return leveled;
}

// ------------------------------------------------------------
// Build gridded height surface from leveled points
// ------------------------------------------------------------
SurfaceGrid buildSurfaceGrid(const std::vector<Point3D>& pts) {
    auto xs = collectAxis(pts, 0);
    auto ys = collectAxis(pts, 1);

    double x_min = percentile(xs, 2.0);
    double x_max = percentile(xs, 98.0);
    double y_min = percentile(ys, 2.0);
    double y_max = percentile(ys, 98.0);

    SurfaceGrid grid;
    grid.res_mm = GRID_RES_MM;
    grid.x_min = x_min;
    grid.y_min = y_min;
    grid.x_center = 0.5 * (x_min + x_max);
    grid.y_center = 0.5 * (y_min + y_max);

    grid.nx = static_cast<int>(std::ceil((x_max - x_min) / GRID_RES_MM)) + 1;
    grid.ny = static_cast<int>(std::ceil((y_max - y_min) / GRID_RES_MM)) + 1;

    grid.h.assign(grid.nx * grid.ny, 0.0);
    grid.valid.assign(grid.nx * grid.ny, 0);

    std::vector<double> sum(grid.nx * grid.ny, 0.0);
    std::vector<int> count(grid.nx * grid.ny, 0);

    for (const auto& p : pts) {
        if (p.x < x_min || p.x > x_max || p.y < y_min || p.y > y_max) continue;

        int ix = static_cast<int>((p.x - x_min) / GRID_RES_MM);
        int iy = static_cast<int>((p.y - y_min) / GRID_RES_MM);

        if (ix < 0 || ix >= grid.nx || iy < 0 || iy >= grid.ny) continue;

        int id = iy * grid.nx + ix;
        sum[id] += p.z;
        count[id]++;
    }

    for (int i = 0; i < grid.nx * grid.ny; ++i) {
        if (count[i] > 0) {
            grid.h[i] = sum[i] / count[i];
            grid.valid[i] = 1;
        }
    }

    // Fill missing cells using neighboring averages.
    for (int iter = 0; iter < 6; ++iter) {
        auto h_new = grid.h;
        auto valid_new = grid.valid;

        for (int y = 1; y < grid.ny - 1; ++y) {
            for (int x = 1; x < grid.nx - 1; ++x) {
                int id = y * grid.nx + x;
                if (grid.valid[id]) continue;

                double total = 0.0;
                int n = 0;

                for (int dy = -1; dy <= 1; ++dy) {
                    for (int dx = -1; dx <= 1; ++dx) {
                        int nid = (y + dy) * grid.nx + (x + dx);
                        if (grid.valid[nid]) {
                            total += grid.h[nid];
                            n++;
                        }
                    }
                }

                if (n > 0) {
                    h_new[id] = total / n;
                    valid_new[id] = 1;
                }
            }
        }

        grid.h = h_new;
        grid.valid = valid_new;
    }

    // Any remaining invalid cells become zero.
    for (int i = 0; i < grid.nx * grid.ny; ++i) {
        if (!grid.valid[i]) {
            grid.h[i] = 0.0;
            grid.valid[i] = 1;
        }
    }

    // Smooth the surface.
    for (int iter = 0; iter < SMOOTH_ITERS; ++iter) {
        auto h_new = grid.h;

        for (int y = 1; y < grid.ny - 1; ++y) {
            for (int x = 1; x < grid.nx - 1; ++x) {
                double total = 0.0;
                int n = 0;

                for (int dy = -1; dy <= 1; ++dy) {
                    for (int dx = -1; dx <= 1; ++dx) {
                        int nid = (y + dy) * grid.nx + (x + dx);
                        total += grid.h[nid];
                        n++;
                    }
                }

                h_new[y * grid.nx + x] = total / n;
            }
        }

        grid.h = h_new;
    }

    grid.h_min = *std::min_element(grid.h.begin(), grid.h.end());
    grid.h_max = *std::max_element(grid.h.begin(), grid.h.end());

    std::cout << "Surface grid: " << grid.nx << " x " << grid.ny << std::endl;
    std::cout << "Height range: " << grid.h_min << " to " << grid.h_max << " mm\n";

    return grid;
}

// ------------------------------------------------------------
// Write outputs
// ------------------------------------------------------------
void writeLeveledPoints(const std::string& path, const std::vector<Point3D>& pts) {
    std::ofstream out(path);

    out << "ply\n";
    out << "format ascii 1.0\n";
    out << "element vertex " << pts.size() << "\n";
    out << "property float x\n";
    out << "property float y\n";
    out << "property float z\n";
    out << "property uchar red\n";
    out << "property uchar green\n";
    out << "property uchar blue\n";
    out << "end_header\n";

    for (const auto& p : pts) {
        out << p.x << " " << p.y << " " << p.z << " "
            << int(p.r) << " " << int(p.g) << " " << int(p.b) << "\n";
    }

    std::cout << "Saved " << path << std::endl;
}

void writeGridCSV(const std::string& path, const SurfaceGrid& grid) {
    std::ofstream out(path);

    for (int y = 0; y < grid.ny; ++y) {
        for (int x = 0; x < grid.nx; ++x) {
            out << grid.h[y * grid.nx + x];
            if (x < grid.nx - 1) out << ",";
        }
        out << "\n";
    }

    std::cout << "Saved " << path << std::endl;
}

void writeSurfaceMeshPLY(const std::string& path, const SurfaceGrid& grid) {
    int vertex_count = grid.nx * grid.ny;
    int face_count = (grid.nx - 1) * (grid.ny - 1) * 2;

    std::ofstream out(path);

    out << "ply\n";
    out << "format ascii 1.0\n";
    out << "element vertex " << vertex_count << "\n";
    out << "property float x\n";
    out << "property float y\n";
    out << "property float z\n";
    out << "element face " << face_count << "\n";
    out << "property list uchar int vertex_indices\n";
    out << "end_header\n";

    for (int y = 0; y < grid.ny; ++y) {
        for (int x = 0; x < grid.nx; ++x) {
            double px = grid.x_min + x * grid.res_mm - grid.x_center;
            double py = grid.y_min + y * grid.res_mm - grid.y_center;
            double pz = grid.h[y * grid.nx + x];

            out << px << " " << py << " " << pz << "\n";
        }
    }

    for (int y = 0; y < grid.ny - 1; ++y) {
        for (int x = 0; x < grid.nx - 1; ++x) {
            int v0 = y * grid.nx + x;
            int v1 = y * grid.nx + (x + 1);
            int v2 = (y + 1) * grid.nx + (x + 1);
            int v3 = (y + 1) * grid.nx + x;

            out << "3 " << v0 << " " << v1 << " " << v2 << "\n";
            out << "3 " << v0 << " " << v2 << " " << v3 << "\n";
        }
    }

    std::cout << "Saved " << path << std::endl;
}

// ------------------------------------------------------------
// OpenGL rendering helpers
// ------------------------------------------------------------
static float rot_x = 55.0f;
static float rot_z = -35.0f;
static float zoom = -6.0f;
static bool wireframe = true;

void renderBitmapString(float x, float y, float z, void* font, const std::string& text) {
    glRasterPos3f(x, y, z);
    for (char c : text) {
        glutBitmapCharacter(font, c);
    }
}

void heightToColor(double h, double h_min, double h_max) {
    double t = (h - h_min) / (h_max - h_min + 1e-9);
    t = std::clamp(t, 0.0, 1.0);

    // Blue/cyan for low, green/yellow/brown for high.
    float r, g, b;

    if (t < 0.25) {
        r = 0.0f;
        g = static_cast<float>(4.0 * t);
        b = 1.0f;
    } else if (t < 0.50) {
        r = 0.0f;
        g = 1.0f;
        b = static_cast<float>(1.0 - 4.0 * (t - 0.25));
    } else if (t < 0.75) {
        r = static_cast<float>(4.0 * (t - 0.50));
        g = 1.0f;
        b = 0.0f;
    } else {
        r = 1.0f;
        g = static_cast<float>(1.0 - 2.0 * (t - 0.75));
        b = static_cast<float>(0.25 * (t - 0.75));
    }

    glColor3f(r, g, b);
}

void drawAxes(float length = 3.0f) {
    glLineWidth(2.0f);
    glBegin(GL_LINES);

    // X axis red
    glColor3f(1, 0, 0);
    glVertex3f(-length, 0, 0);
    glVertex3f(length, 0, 0);

    // Y axis green
    glColor3f(0, 1, 0);
    glVertex3f(0, -length, 0);
    glVertex3f(0, length, 0);

    // Z axis blue
    glColor3f(0, 0, 1);
    glVertex3f(0, 0, -1.0f);
    glVertex3f(0, 0, 1.0f);

    glEnd();

    glColor3f(1, 0, 0);
    renderBitmapString(length, 0, 0, GLUT_BITMAP_HELVETICA_12, "X");

    glColor3f(0, 1, 0);
    renderBitmapString(0, length, 0, GLUT_BITMAP_HELVETICA_12, "Y");

    glColor3f(0, 0, 1);
    renderBitmapString(0, 0, 1.0f, GLUT_BITMAP_HELVETICA_12, "Height");
}

void drawSurface(const SurfaceGrid& grid) {
    glPolygonMode(GL_FRONT_AND_BACK, wireframe ? GL_LINE : GL_FILL);
    glLineWidth(wireframe ? 1.0f : 0.5f);

    glBegin(GL_QUADS);

    for (int y = 0; y < grid.ny - 1; ++y) {
        for (int x = 0; x < grid.nx - 1; ++x) {
            int id0 = y * grid.nx + x;
            int id1 = y * grid.nx + (x + 1);
            int id2 = (y + 1) * grid.nx + (x + 1);
            int id3 = (y + 1) * grid.nx + x;

            double px0 = (grid.x_min + x * grid.res_mm - grid.x_center) / 1000.0;
            double py0 = (grid.y_min + y * grid.res_mm - grid.y_center) / 1000.0;

            double px1 = (grid.x_min + (x + 1) * grid.res_mm - grid.x_center) / 1000.0;
            double py1 = py0;

            double px2 = px1;
            double py2 = (grid.y_min + (y + 1) * grid.res_mm - grid.y_center) / 1000.0;

            double px3 = px0;
            double py3 = py2;

            double z0 = (grid.h[id0] / 1000.0) * HEIGHT_EXAGGERATION;
            double z1 = (grid.h[id1] / 1000.0) * HEIGHT_EXAGGERATION;
            double z2 = (grid.h[id2] / 1000.0) * HEIGHT_EXAGGERATION;
            double z3 = (grid.h[id3] / 1000.0) * HEIGHT_EXAGGERATION;

            heightToColor(grid.h[id0], grid.h_min, grid.h_max);
            glVertex3f(px0, py0, z0);

            heightToColor(grid.h[id1], grid.h_min, grid.h_max);
            glVertex3f(px1, py1, z1);

            heightToColor(grid.h[id2], grid.h_min, grid.h_max);
            glVertex3f(px2, py2, z2);

            heightToColor(grid.h[id3], grid.h_min, grid.h_max);
            glVertex3f(px3, py3, z3);
        }
    }

    glEnd();

    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
}

void processInput(GLFWwindow* window) {
    if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS ||
        glfwGetKey(window, GLFW_KEY_Q) == GLFW_PRESS) {
        glfwSetWindowShouldClose(window, true);
    }

    if (glfwGetKey(window, GLFW_KEY_LEFT) == GLFW_PRESS)  rot_z -= 1.5f;
    if (glfwGetKey(window, GLFW_KEY_RIGHT) == GLFW_PRESS) rot_z += 1.5f;
    if (glfwGetKey(window, GLFW_KEY_UP) == GLFW_PRESS)    rot_x -= 1.5f;
    if (glfwGetKey(window, GLFW_KEY_DOWN) == GLFW_PRESS)  rot_x += 1.5f;

    if (glfwGetKey(window, GLFW_KEY_EQUAL) == GLFW_PRESS) zoom += 0.05f;
    if (glfwGetKey(window, GLFW_KEY_MINUS) == GLFW_PRESS) zoom -= 0.05f;
}

void keyCallback(GLFWwindow* window, int key, int, int action, int) {
    if (action == GLFW_PRESS && key == GLFW_KEY_W) {
        wireframe = !wireframe;
        std::cout << "Wireframe: " << (wireframe ? "ON" : "OFF") << std::endl;
    }
}



// ------------------------------------------------------------
// Main
// ------------------------------------------------------------
int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: ./road_surface_model_viewer <input.ply>\n";
        return 1;
    }

    std::string ply_path = argv[1];

    std::vector<Point3D> points;
    if (!readAsciiPly(ply_path, points)) {
        return 1;
    }

  

    // Remove bottom 20% first to avoid the hood contaminating plane fitting.
    auto noBottom = removeBottomPercentOfCloud(points);

    auto downsampled = voxelDownsample(noBottom, VOXEL_SIZE_MM);

    // Then do the regular percentile crop.
    auto cropped = cropPercentile(downsampled);

    Eigen::Vector3d normal;
    Eigen::Vector3d plane_center;
    double d = 0.0;

    if (!fitPlaneRANSAC(cropped, normal, d, plane_center)) {
        return 1;
    }

    auto leveled = levelPoints(cropped, normal, plane_center);
    auto grid = buildSurfaceGrid(leveled);

    writeLeveledPoints("leveled_points.ply", leveled);
    writeSurfaceMeshPLY("surface_model_grid.ply", grid);
    writeGridCSV("surface_grid.csv", grid);

    // OpenGL viewer
    if (!glfwInit()) {
        std::cerr << "GLFW init failed.\n";
        return 1;
    }

    GLFWwindow* window = glfwCreateWindow(
        1100, 850,
        "Road Surface Model - PLY RANSAC Leveling",
        nullptr,
        nullptr
    );

    if (!window) {
        std::cerr << "Window creation failed.\n";
        glfwTerminate();
        return 1;
    }

    glfwMakeContextCurrent(window);
    glfwSetKeyCallback(window, keyCallback);

    if (glewInit() != GLEW_OK) {
        std::cerr << "GLEW init failed.\n";
        glfwTerminate();
        return 1;
    }

    glutInit(&argc, argv);

    glEnable(GL_DEPTH_TEST);
    glClearColor(0.02f, 0.02f, 0.025f, 1.0f);

    std::cout << "\nControls:\n";
    std::cout << "  Arrow keys: rotate view\n";
    std::cout << "  +/-       : zoom\n";
    std::cout << "  W         : toggle wireframe/fill\n";
    std::cout << "  Q or ESC  : quit\n\n";

    while (!glfwWindowShouldClose(window)) {
        processInput(window);

        int width, height;
        glfwGetFramebufferSize(window, &width, &height);

        glViewport(0, 0, width, height);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        glMatrixMode(GL_PROJECTION);
        glLoadIdentity();
        gluPerspective(45.0, double(width) / double(height), 0.01, 100.0);

        glMatrixMode(GL_MODELVIEW);
        glLoadIdentity();

        glTranslatef(0.0f, 0.0f, zoom);
        glRotatef(rot_x, 1.0f, 0.0f, 0.0f);
        glRotatef(rot_z, 0.0f, 0.0f, 1.0f);

        drawAxes(3.0f);
        drawSurface(grid);

        glColor3f(0.0f, 1.0f, 0.0f);
        renderBitmapString(
            1.8f, 2.0f, 0.8f,
            GLUT_BITMAP_HELVETICA_18,
            "Leveled Road Surface Model"
        );

        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    glfwDestroyWindow(window);
    glfwTerminate();

    return 0;
}