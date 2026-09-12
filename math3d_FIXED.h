// ============================================================================
// math3d.h — Core math library for SpaceGame
// Double-precision world vectors (Vec3d) enable the Floating Origin system
// across a >1e12 unit universe. Float Mat4/Quat used for rendering.
// ============================================================================
#pragma once
#include <cmath>
#include <cstdint>
#include <algorithm>

namespace sg {

constexpr float  PI  = 3.14159265358979323846f;
constexpr float  TAU = 6.28318530717958647692f;
constexpr double PI_D = 3.14159265358979323846;

inline float clampf(float v, float lo, float hi) { return v < lo ? lo : (v > hi ? hi : v); }
inline float lerpf(float a, float b, float t) { return a + (b - a) * t; }
inline float smoothstep01(float t) { t = clampf(t, 0.f, 1.f); return t * t * (3.f - 2.f * t); }
inline float deg2rad(float d) { return d * (PI / 180.f); }
inline float rad2deg(float r) { return r * (180.f / PI); }

// ----------------------------------------------------------------------------
struct Vec2 {
    float x = 0, y = 0;
    Vec2() {}
    Vec2(float a, float b) : x(a), y(b) {}
    Vec2 operator+(const Vec2& o) const { return { x + o.x, y + o.y }; }
    Vec2 operator-(const Vec2& o) const { return { x - o.x, y - o.y }; }
    Vec2 operator*(float s) const { return { x * s, y * s }; }
    Vec2& operator+=(const Vec2& o) { x += o.x; y += o.y; return *this; }
    float len() const { return std::sqrt(x * x + y * y); }
};

struct Vec3 {
    float x = 0, y = 0, z = 0;
    Vec3() {}
    Vec3(float a) : x(a), y(a), z(a) {}
    Vec3(float a, float b, float c) : x(a), y(b), z(c) {}
    Vec3 operator+(const Vec3& o) const { return { x + o.x, y + o.y, z + o.z }; }
    Vec3 operator-(const Vec3& o) const { return { x - o.x, y - o.y, z - o.z }; }
    Vec3 operator*(float s) const { return { x * s, y * s, z * s }; }
    Vec3 operator/(float s) const { float i = 1.f / s; return { x * i, y * i, z * i }; }
    Vec3 operator*(const Vec3& o) const { return { x * o.x, y * o.y, z * o.z }; }
    Vec3 operator-() const { return { -x, -y, -z }; }
    Vec3& operator+=(const Vec3& o) { x += o.x; y += o.y; z += o.z; return *this; }
    Vec3& operator-=(const Vec3& o) { x -= o.x; y -= o.y; z -= o.z; return *this; }
    Vec3& operator*=(float s) { x *= s; y *= s; z *= s; return *this; }
    float dot(const Vec3& o) const { return x * o.x + y * o.y + z * o.z; }
    Vec3 cross(const Vec3& o) const {
        return { y * o.z - z * o.y, z * o.x - x * o.z, x * o.y - y * o.x };
    }
    float lenSq() const { return dot(*this); }
    float len() const { return std::sqrt(lenSq()); }
    Vec3 norm() const { float l = len(); return l > 1e-20f ? (*this) / l : Vec3(0, 0, 1); }
    bool isFinite() const { return std::isfinite(x) && std::isfinite(y) && std::isfinite(z); }
    static Vec3 up() { return { 0, 1, 0 }; }
    static Vec3 zero() { return { 0, 0, 0 }; }
};
inline Vec3 operator*(float s, const Vec3& v) { return v * s; }

// Double precision world position
struct Vec3d {
    double x = 0, y = 0, z = 0;
    Vec3d() {}
    Vec3d(double a) : x(a), y(a), z(a) {}
    Vec3d(double a, double b, double c) : x(a), y(b), z(c) {}
    explicit Vec3d(const Vec3& v) : x(v.x), y(v.y), z(v.z) {}
    Vec3d operator+(const Vec3d& o) const { return { x + o.x, y + o.y, z + o.z }; }
    Vec3d operator-(const Vec3d& o) const { return { x - o.x, y - o.y, z - o.z }; }
    Vec3d operator*(double s) const { return { x * s, y * s, z * s }; }
    Vec3d operator/(double s) const { double i = 1.0 / s; return { x * i, y * i, z * i }; }
    Vec3d operator-() const { return { -x, -y, -z }; }
    Vec3d& operator+=(const Vec3d& o) { x += o.x; y += o.y; z += o.z; return *this; }
    Vec3d& operator-=(const Vec3d& o) { x -= o.x; y -= o.y; z -= o.z; return *this; }
    double dot(const Vec3d& o) const { return x * o.x + y * o.y + z * o.z; }
    Vec3d cross(const Vec3d& o) const {
        return { y * o.z - z * o.y, z * o.x - x * o.z, x * o.y - y * o.x };
    }
    double lenSq() const { return dot(*this); }
    double len() const { return std::sqrt(lenSq()); }
    Vec3d norm() const { double l = len(); return l > 1e-30 ? (*this) / l : Vec3d(0, 0, 1); }
    bool isFinite() const { return std::isfinite(x) && std::isfinite(y) && std::isfinite(z); }
    Vec3 toFloat() const { return { (float)x, (float)y, (float)z }; }
    // Camera-relative conversion: the heart of Floating Origin.
    Vec3 relativeTo(const Vec3d& origin) const { return (*this - origin).toFloat(); }
    bool nearEqual(const Vec3d& o, double eps) const { return (*this - o).lenSq() < eps * eps; }
};
inline Vec3d operator*(double s, const Vec3d& v) { return v * s; }

struct Vec4 {
    float x = 0, y = 0, z = 0, w = 0;
    Vec4() {}
    Vec4(float a, float b, float c, float d) : x(a), y(b), z(c), w(d) {}
    Vec4(const Vec3& v, float d) : x(v.x), y(v.y), z(v.z), w(d) {}
};

// ----------------------------------------------------------------------------
// Quaternion (float) — ship orientation
struct Quat {
    float x = 0, y = 0, z = 0, w = 1;
    Quat() {}
    Quat(float a, float b, float c, float d) : x(a), y(b), z(c), w(d) {}

    static Quat identity() { return { 0, 0, 0, 1 }; }
    static Quat fromAxisAngle(const Vec3& axis, float ang) {
        Vec3 a = axis.norm();
        float h = ang * 0.5f, s = std::sin(h);
        return { a.x * s, a.y * s, a.z * s, std::cos(h) };
    }
    Quat operator*(const Quat& o) const {
        return {
            w * o.x + x * o.w + y * o.z - z * o.y,
            w * o.y - x * o.z + y * o.w + z * o.x,
            w * o.z + x * o.y - y * o.x + z * o.w,
            w * o.w - x * o.x - y * o.y - z * o.z };
    }
    Quat conj() const { return { -x, -y, -z, w }; }
    float normSq() const { return x * x + y * y + z * z + w * w; }
    Quat norm() const {
        float l = std::sqrt(normSq());
        if (l < 1e-20f) return identity();
        float i = 1.f / l;
        return { x * i, y * i, z * i, w * i };
    }
    Vec3 rotate(const Vec3& v) const {
        // q * v * q^-1 optimized
        Vec3 qv(x, y, z);
        Vec3 t = 2.f * qv.cross(v);
        return v + w * t + qv.cross(t);
    }
    Vec3 forward() const { return rotate(Vec3(0, 0, -1)); }
    Vec3 right()   const { return rotate(Vec3(1, 0, 0)); }
    Vec3 up()      const { return rotate(Vec3(0, 1, 0)); }
    bool isFinite() const { return std::isfinite(x) && std::isfinite(y) && std::isfinite(z) && std::isfinite(w); }

    static Quat slerp(const Quat& a, const Quat& b, float t) {
        float cosom = a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
        Quat bb = b;
        if (cosom < 0.f) { cosom = -cosom; bb = { -b.x, -b.y, -b.z, -b.w }; }
        if (cosom > 0.9995f) {
            return Quat(a.x + t * (bb.x - a.x), a.y + t * (bb.y - a.y),
                        a.z + t * (bb.z - a.z), a.w + t * (bb.w - a.w)).norm();
        }
        float theta = std::acos(clampf(cosom, -1.f, 1.f));
        float sinom = std::sin(theta);
        float wa = std::sin((1.f - t) * theta) / sinom;
        float wb = std::sin(t * theta) / sinom;
        return Quat(wa * a.x + wb * bb.x, wa * a.y + wb * bb.y,
                    wa * a.z + wb * bb.z, wa * a.w + wb * bb.w).norm();
    }
    // Small incremental rotation from angular velocity (rad/s) over dt
    Quat integrate(const Vec3& angVel, float dt) const {
        float w2 = angVel.len();
        if (w2 < 1e-6f) return *this;
        Quat dq = fromAxisAngle(angVel / w2, w2 * dt);
        return (dq * (*this)).norm();
    }
};

// ----------------------------------------------------------------------------
// 4x4 matrix, column-major (OpenGL convention): m[col*4+row]
struct Mat4 {
    float m[16];
    // NOTE: must NOT call identity() here — identity() used to construct a
    // local `Mat4 r;`, which re-invoked this constructor and recursed
    // infinitely (stack overflow crash at startup). Build the identity
    // matrix in-place instead.
    Mat4() { for (int i = 0; i < 16; i++) m[i] = 0.f; m[0] = m[5] = m[10] = m[15] = 1.f; }
    static Mat4 identity() { return Mat4(); }
    static Mat4 translate(const Vec3& t) {
        Mat4 r = identity();
        r.m[12] = t.x; r.m[13] = t.y; r.m[14] = t.z;
        return r;
    }
    static Mat4 scale(const Vec3& s) {
        Mat4 r = identity();
        r.m[0] = s.x; r.m[5] = s.y; r.m[10] = s.z;
        return r;
    }
    static Mat4 fromQuat(const Quat& q) {
        float xx = q.x * q.x, yy = q.y * q.y, zz = q.z * q.z;
        float xy = q.x * q.y, xz = q.x * q.z, yz = q.y * q.z;
        float wx = q.w * q.x, wy = q.w * q.y, wz = q.w * q.z;
        Mat4 r;
        r.m[0] = 1 - 2 * (yy + zz); r.m[1] = 2 * (xy + wz);     r.m[2] = 2 * (xz - wy);     r.m[3] = 0;
        r.m[4] = 2 * (xy - wz);     r.m[5] = 1 - 2 * (xx + zz); r.m[6] = 2 * (yz + wx);     r.m[7] = 0;
        r.m[8] = 2 * (xz + wy);     r.m[9] = 2 * (yz - wx);     r.m[10] = 1 - 2 * (xx + yy);r.m[11] = 0;
        r.m[12] = 0; r.m[13] = 0; r.m[14] = 0; r.m[15] = 1;
        return r;
    }
    static Mat4 TRS(const Vec3& t, const Quat& rot, const Vec3& s) {
        Mat4 r = fromQuat(rot);
        r.m[0] *= s.x; r.m[1] *= s.x; r.m[2] *= s.x;
        r.m[4] *= s.y; r.m[5] *= s.y; r.m[6] *= s.y;
        r.m[8] *= s.z; r.m[9] *= s.z; r.m[10] *= s.z;
        r.m[12] = t.x; r.m[13] = t.y; r.m[14] = t.z;
        return r;
    }
    Mat4 operator*(const Mat4& o) const {
        Mat4 r;
        for (int c = 0; c < 4; c++)
            for (int row = 0; row < 4; row++) {
                float s = 0;
                for (int k = 0; k < 4; k++) s += m[k * 4 + row] * o.m[c * 4 + k];
                r.m[c * 4 + row] = s;
            }
        return r;
    }
    static Mat4 perspective(float fovYRad, float aspect, float zn, float zf) {
        Mat4 r;
        for (int i = 0; i < 16; i++) r.m[i] = 0.f;
        float f = 1.f / std::tan(fovYRad * 0.5f);
        r.m[0] = f / aspect;
        r.m[5] = f;
        r.m[10] = (zf + zn) / (zn - zf);
        r.m[11] = -1.f;
        r.m[14] = (2.f * zf * zn) / (zn - zf);
        return r;
    }
    static Mat4 ortho(float l, float rr, float b, float t, float n, float f) {
        Mat4 r = identity();
        r.m[0] = 2.f / (rr - l);
        r.m[5] = 2.f / (t - b);
        r.m[10] = -2.f / (f - n);
        r.m[12] = -(rr + l) / (rr - l);
        r.m[13] = -(t + b) / (t - b);
        r.m[14] = -(f + n) / (f - n);
        return r;
    }
    // View matrix from camera basis vectors (camera at relative-space origin).
    // Rotation-only view: rows are the camera basis (Floating Origin pattern).
    static Mat4 viewFromBasis(const Vec3& fwd, const Vec3& up, const Vec3& right) {
        Mat4 r = identity();
        r.m[0] = right.x; r.m[4] = right.y; r.m[8]  = right.z;
        r.m[1] = up.x;    r.m[5] = up.y;    r.m[9]  = up.z;
        r.m[2] = -fwd.x;  r.m[6] = -fwd.y;  r.m[10] = -fwd.z;
        r.m[12] = 0; r.m[13] = 0; r.m[14] = 0;
        return r;
    }
    Vec3 transformPoint(const Vec3& p) const {
        float w = m[3] * p.x + m[7] * p.y + m[11] * p.z + m[15];
        if (std::fabs(w) < 1e-20f) w = 1e-20f;
        float iw = 1.f / w;
        return { (m[0] * p.x + m[4] * p.y + m[8]  * p.z + m[12]) * iw,
                 (m[1] * p.x + m[5] * p.y + m[9]  * p.z + m[13]) * iw,
                 (m[2] * p.x + m[6] * p.y + m[10] * p.z + m[14]) * iw };
    }
    Vec3 transformDir(const Vec3& p) const {
        return { m[0] * p.x + m[4] * p.y + m[8]  * p.z,
                 m[1] * p.x + m[5] * p.y + m[9]  * p.z,
                 m[2] * p.x + m[6] * p.y + m[10] * p.z };
    }
    Mat4 invView() const {
        // For rotation+translation view matrices
        Mat4 r = identity();
        r.m[0] = m[0]; r.m[1] = m[4]; r.m[2] = m[8];
        r.m[4] = m[1]; r.m[5] = m[5]; r.m[6] = m[9];
        r.m[8] = m[2]; r.m[9] = m[6]; r.m[10] = m[10];
        Vec3 t(m[12], m[13], m[14]);
        r.m[12] = -(r.m[0] * t.x + r.m[4] * t.y + r.m[8] * t.z);
        r.m[13] = -(r.m[1] * t.x + r.m[5] * t.y + r.m[9] * t.z);
        r.m[14] = -(r.m[2] * t.x + r.m[6] * t.y + r.m[10] * t.z);
        return r;
    }
};

// ----------------------------------------------------------------------------
// Projection helper: world-rel point -> screen [0..1] with z. w<=0 => behind.
struct ProjectResult { float x, y, z, w; bool visible; };
inline ProjectResult projectPoint(const Mat4& viewProj, const Vec3& relPos, int screenW, int screenH) {
    ProjectResult r{ 0, 0, 0, 0, false };
    const float* m = viewProj.m;
    float cx = m[0] * relPos.x + m[4] * relPos.y + m[8]  * relPos.z + m[12];
    float cy = m[1] * relPos.x + m[5] * relPos.y + m[9]  * relPos.z + m[13];
    float cz = m[2] * relPos.x + m[6] * relPos.y + m[10] * relPos.z + m[14];
    float cw = m[3] * relPos.x + m[7] * relPos.y + m[11] * relPos.z + m[15];
    r.w = cw;
    if (cw > 1e-6f) {
        float iw = 1.f / cw;
        r.x = (cx * iw * 0.5f + 0.5f) * (float)screenW;
        r.y = (1.f - (cy * iw * 0.5f + 0.5f)) * (float)screenH;
        r.z = cz * iw;
        r.visible = (r.z > -1.001f && r.z < 1.001f);
    } else {
        // Behind camera: project to the mirrored position so brackets flip properly
        float iw = 1.f / std::max(1e-6f, -cw);
        r.x = (1.f - (cx * iw * 0.5f + 0.5f)) * (float)screenW;
        r.y = (1.f - (cy * iw * 0.5f + 0.5f)) * (float)screenH;
    }
    return r;
}

// Format huge distances: 5.23e+11 -> "523.0 Gu" (giga-units) style for HUD
inline void formatDistance(double d, char* out, int outLen) {
    if (!std::isfinite(d)) { snprintf(out, outLen, "---"); return; }
    double a = std::fabs(d);
    if (a < 1e3)       snprintf(out, outLen, "%.0f u", d);
    else if (a < 1e6)  snprintf(out, outLen, "%.2f ku", d / 1e3);
    else if (a < 1e9)  snprintf(out, outLen, "%.2f Mu", d / 1e6);
    else if (a < 1e12) snprintf(out, outLen, "%.2f Gu", d / 1e9);
    else               snprintf(out, outLen, "%.2f Tu", d / 1e12);
}

} // namespace sg
