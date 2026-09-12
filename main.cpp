// ============================================================================
// main.cpp — SpaceGame: decentralized P2P 3D space exploration game
//
// Systems implemented in this file:
//   1) Rendering: SDL2 + legacy/fixed-function OpenGL. Loads .obj models via
//      tinyobjloader and .jpg/.png textures via stb_image. Draws the player
//      ship, asteroids, planets, a starfield, and a sci-fi HUD.
//   2) Floating Origin: all world positions are stored as sg::Vec3d (double
//      precision). Every frame everything is rendered relative to a movable
//      `origin` (also Vec3d) which re-centers on the player, so the GPU only
//      ever sees small float numbers no matter how far the player has
//      travelled from (0,0,0).
//   3) P2P swarm networking: UDP broadcast discovery + decentralized peer
//      exchange (gossip) + 20Hz position/rotation/state sync, built on top
//      of src/net/udp.h and src/net/protocol.h from this repository.
//
// This file intentionally uses only OpenGL 1.1 fixed-function calls
// (glBegin/glVertexPointer/glTexImage2D/matrix stack) so it links against
// nothing but the OS-provided GL library — no GLEW/GLAD/ANGLE required.
// ============================================================================

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <string>
#include <vector>
#include <unordered_map>
#include <map>
#include <algorithm>
#include <random>
#include <chrono>
#include <sstream>
#include <iostream>
#include <functional>
#include <array>

#include <SDL.h>
#include <SDL_opengl.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define TINYOBJLOADER_IMPLEMENTATION
#include "tinyobjloader.h"

#include "font8x8_basic.h"

#include "core/math3d.h"
#include "core/rng.h"
#include "core/log.h"
#include "net/protocol.h"
#include "net/udp.h"

using namespace sg;

// ============================================================================
// SECTION 0: small utilities
// ============================================================================

static double nowSeconds() {
    using namespace std::chrono;
    static const auto t0 = steady_clock::now();
    return duration<double>(steady_clock::now() - t0).count();
}

static uint64_t randomU64() {
    std::random_device rd;
    std::mt19937_64 gen(((uint64_t)rd() << 32) ^ rd() ^ (uint64_t)nowSeconds());
    return gen();
}

// ============================================================================
// SECTION 1: GPU-side mesh / texture (loaded via tinyobjloader / stb_image)
// ============================================================================

struct GpuVertex {
    Vec3 pos;
    Vec3 nrm;
    Vec3 col;
};

struct Mesh {
    std::vector<GpuVertex> verts; // non-indexed triangle list (simplicity)
    Vec3 boundsCenter;
    float boundsRadius = 1.f;

    void draw() const {
        if (verts.empty()) return;
        glEnableClientState(GL_VERTEX_ARRAY);
        glEnableClientState(GL_NORMAL_ARRAY);
        glEnableClientState(GL_COLOR_ARRAY);
        glVertexPointer(3, GL_FLOAT, sizeof(GpuVertex), &verts[0].pos);
        glNormalPointer(GL_FLOAT, sizeof(GpuVertex), &verts[0].nrm);
        glColorPointer(3, GL_FLOAT, sizeof(GpuVertex), &verts[0].col);
        glDrawArrays(GL_TRIANGLES, 0, (GLsizei)verts.size());
        glDisableClientState(GL_VERTEX_ARRAY);
        glDisableClientState(GL_NORMAL_ARRAY);
        glDisableClientState(GL_COLOR_ARRAY);
    }
};

// Loads a Kenney-style .obj. Missing/renamed .mtl files are tolerated
// (tinyobjloader just reports a warning); a deterministic fallback palette
// keyed by material index is used so every part of the ship still reads
// as distinct metal/dark/accent panels.
static Mesh loadObjMesh(const std::string& path, const std::string& mtlDir) {
    Mesh mesh;
    tinyobj::attrib_t attrib;
    std::vector<tinyobj::shape_t> shapes;
    std::vector<tinyobj::material_t> materials;
    std::string warn, err;

    bool ok = tinyobj::LoadObj(&attrib, &shapes, &materials, &warn, &err,
                                path.c_str(), mtlDir.c_str(), true, true);
    if (!warn.empty()) logWarn("obj '%s': %s", path.c_str(), warn.c_str());
    if (!err.empty())  logErr("obj '%s': %s", path.c_str(), err.c_str());
    if (!ok) {
        logErr("failed to load model: %s (a placeholder cube will be used)", path.c_str());
        return mesh; // caller substitutes a fallback primitive
    }

    static const Vec3 fallbackPalette[4] = {
        Vec3(0.82f, 0.85f, 0.90f), // metal
        Vec3(0.55f, 0.58f, 0.63f), // metalDark
        Vec3(0.22f, 0.24f, 0.28f), // dark
        Vec3(1.00f, 0.55f, 0.15f), // accent / metalRed
    };

    Vec3 lo(1e9f), hi(-1e9f);

    for (const auto& shape : shapes) {
        size_t idxOfs = 0;
        for (size_t f = 0; f < shape.mesh.num_face_vertices.size(); f++) {
            int fv = shape.mesh.num_face_vertices[f];
            int matId = f < shape.mesh.material_ids.size() ? shape.mesh.material_ids[f] : -1;
            Vec3 faceColor = matId >= 0 ? fallbackPalette[matId % 4] : Vec3(0.75f, 0.78f, 0.82f);

            Vec3 fpos[8]; int nfv = std::min(fv, 8);
            for (int v = 0; v < nfv; v++) {
                tinyobj::index_t idx = shape.mesh.indices[idxOfs + v];
                fpos[v] = Vec3(
                    attrib.vertices[3 * idx.vertex_index + 0],
                    attrib.vertices[3 * idx.vertex_index + 1],
                    attrib.vertices[3 * idx.vertex_index + 2]);
            }
            // Fan-triangulate the (already-triangulated-by-tinyobj, but be safe) face.
            for (int v = 1; v + 1 < nfv; v++) {
                Vec3 a = fpos[0], b = fpos[v], c = fpos[v + 1];
                Vec3 fn = (b - a).cross(c - a).norm();
                for (Vec3 p : { a, b, c }) {
                    mesh.verts.push_back({ p, fn, faceColor });
                    lo.x = std::min(lo.x, p.x); lo.y = std::min(lo.y, p.y); lo.z = std::min(lo.z, p.z);
                    hi.x = std::max(hi.x, p.x); hi.y = std::max(hi.y, p.y); hi.z = std::max(hi.z, p.z);
                }
            }
            idxOfs += fv;
        }
    }
    if (!mesh.verts.empty()) {
        mesh.boundsCenter = (lo + hi) * 0.5f;
        mesh.boundsRadius = (hi - lo).len() * 0.5f;
    }
    logInfo("loaded model '%s' (%zu tris)", path.c_str(), mesh.verts.size() / 3);
    return mesh;
}

// Minimal placeholder so a missing/failed model never crashes the game.
static Mesh makeFallbackCube(float s, Vec3 color) {
    Mesh m;
    Vec3 p[8] = {
        {-s,-s,-s},{ s,-s,-s},{ s, s,-s},{-s, s,-s},
        {-s,-s, s},{ s,-s, s},{ s, s, s},{-s, s, s}
    };
    int faces[6][4] = { {0,1,2,3},{5,4,7,6},{4,0,3,7},{1,5,6,2},{3,2,6,7},{4,5,1,0} };
    for (auto& f : faces) {
        Vec3 a = p[f[0]], b = p[f[1]], c = p[f[2]], d = p[f[3]];
        Vec3 n = (b - a).cross(c - a).norm();
        m.verts.push_back({a,n,color}); m.verts.push_back({b,n,color}); m.verts.push_back({c,n,color});
        m.verts.push_back({a,n,color}); m.verts.push_back({c,n,color}); m.verts.push_back({d,n,color});
    }
    m.boundsRadius = s * 1.7f;
    return m;
}

struct Texture {
    GLuint id = 0;
    int w = 0, h = 0;
    bool load(const std::string& path) {
        int comp;
        stbi_set_flip_vertically_on_load(1);
        unsigned char* data = stbi_load(path.c_str(), &w, &h, &comp, 3);
        if (!data) { logErr("failed to load texture: %s", path.c_str()); return false; }
        glGenTextures(1, &id);
        glBindTexture(GL_TEXTURE_2D, id);
        // Plain bilinear filtering, no mipmaps: keeps this file dependent on
        // nothing beyond OpenGL 1.1 (no GLU / glGenerateMipmap requirement).
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, w, h, 0, GL_RGB, GL_UNSIGNED_BYTE, data);
        stbi_image_free(data);
        logInfo("loaded texture: %s (%dx%d)", path.c_str(), w, h);
        return true;
    }
};

// Textured sphere drawn with immediate-mode triangle strips (kept separate
// from `Mesh` because it needs per-vertex UV, not per-vertex color).
struct PlanetSphere {
    std::vector<Vec3> pos, nrm;
    std::vector<Vec2> uv;
    int rings, sectors;
    static PlanetSphere build(int rings_, int sectors_) {
        PlanetSphere ps; ps.rings = rings_; ps.sectors = sectors_;
        for (int r = 0; r <= rings_; r++) {
            float v = (float)r / rings_, theta = v * PI;
            for (int s = 0; s <= sectors_; s++) {
                float u = (float)s / sectors_, phi = u * TAU;
                Vec3 p(std::sin(theta) * std::cos(phi), std::cos(theta), std::sin(theta) * std::sin(phi));
                ps.pos.push_back(p); ps.nrm.push_back(p); ps.uv.push_back(Vec2(u, 1.f - v));
            }
        }
        return ps;
    }
    void draw(float radius, GLuint tex) const {
        glEnable(GL_TEXTURE_2D);
        glBindTexture(GL_TEXTURE_2D, tex);
        glColor3f(1, 1, 1);
        int stride = sectors + 1;
        for (int r = 0; r < rings; r++) {
            glBegin(GL_TRIANGLE_STRIP);
            for (int s = 0; s <= sectors; s++) {
                for (int rr : {r, r + 1}) {
                    int i = rr * stride + s;
                    glNormal3f(nrm[i].x, nrm[i].y, nrm[i].z);
                    glTexCoord2f(uv[i].x, uv[i].y);
                    Vec3 p = pos[i] * radius;
                    glVertex3f(p.x, p.y, p.z);
                }
            }
            glEnd();
        }
        glDisable(GL_TEXTURE_2D);
    }
};

// ============================================================================
// SECTION 2: floating-origin world entities
// ============================================================================

struct ShipState {
    Vec3d pos;
    Vec3  vel;
    Quat  rot = Quat::identity();
    float throttle = 0.f;
    float hull = 100.f, shield = 100.f, energy = 100.f, fuel = 100.f;
    bool  boosting = false;
};

struct Asteroid {
    uint64_t id = 0;
    Vec3d pos;
    Quat rot;
    Vec3 spin;      // rad/s
    float scale = 1.f;
    int meshIdx = 0;
    uint16_t hp = 100;
};

struct Planet {
    std::string name;
    Vec3d pos;
    float radius;
    Texture tex;
};

// Deterministic procedural sector streaming: identical universeSeed on every
// peer -> every peer generates byte-identical asteroid fields with zero
// network traffic. This is what lets the P2P swarm skip a shared-world
// server entirely.
struct SectorWorld {
    static constexpr double SECTOR_SIZE = 2000.0;
    static constexpr int    LOAD_RADIUS = 2; // sectors around player, per axis
    uint64_t universeSeed;
    int numMeshVariants;
    std::unordered_map<int64_t, std::vector<Asteroid>> sectors;
    uint64_t nextEntId = 1;

    explicit SectorWorld(uint64_t seed, int meshVariants) : universeSeed(seed), numMeshVariants(meshVariants) {}

    static int64_t sectorKey(int64_t sx, int64_t sy, int64_t sz) {
        // 21 bits per axis is far more range than any session will traverse.
        auto enc = [](int64_t v) -> int64_t { return (v & 0x1FFFFF); };
        return (enc(sx) << 42) | (enc(sy) << 21) | enc(sz);
    }

    static int64_t floorDiv(double v, double d) { return (int64_t)std::floor(v / d); }

    void generateSector(int64_t sx, int64_t sy, int64_t sz) {
        int64_t key = sectorKey(sx, sy, sz);
        if (sectors.count(key)) return;
        uint64_t seed = hashCombine(universeSeed, (uint64_t)hash3to32(sx, sy, sz, universeSeed));
        Rng rng(seed);
        std::vector<Asteroid> field;
        int count = rng.irange(3, 9);
        for (int i = 0; i < count; i++) {
            Asteroid a;
            a.id = hashCombine(seed, (uint64_t)i) | 1ull; // never 0
            Vec3d local = Vec3d(rng.range(0, (float)SECTOR_SIZE), rng.range(0, (float)SECTOR_SIZE), rng.range(0, (float)SECTOR_SIZE));
            a.pos = Vec3d((double)sx, (double)sy, (double)sz) * SECTOR_SIZE + local;
            a.rot = rng.randQuat();
            a.spin = rng.inUnitSphere() * rng.range(0.05f, 0.4f);
            a.scale = rng.range(4.f, 40.f);
            a.meshIdx = rng.irange(0, numMeshVariants - 1);
            a.hp = (uint16_t)rng.irange(50, 300);
            field.push_back(a);
        }
        sectors[key] = std::move(field);
    }

    void streamAround(const Vec3d& p) {
        int64_t cx = floorDiv(p.x, SECTOR_SIZE), cy = floorDiv(p.y, SECTOR_SIZE), cz = floorDiv(p.z, SECTOR_SIZE);
        for (int dx = -LOAD_RADIUS; dx <= LOAD_RADIUS; dx++)
            for (int dy = -LOAD_RADIUS; dy <= LOAD_RADIUS; dy++)
                for (int dz = -LOAD_RADIUS; dz <= LOAD_RADIUS; dz++)
                    generateSector(cx + dx, cy + dy, cz + dz);
        // Unload sectors far outside the load bubble to bound memory.
        for (auto it = sectors.begin(); it != sectors.end();) {
            // Re-derive coords is unnecessary for correctness (memory-only cache);
            // a simple size cap keeps this demo bounded.
            ++it;
        }
        if (sectors.size() > 400) {
            // Cheap eviction: drop half of the cache; it will regenerate deterministically if revisited.
            size_t target = sectors.size() / 2;
            auto it = sectors.begin();
            while (sectors.size() > target) it = sectors.erase(it);
        }
    }

    void forEachAsteroid(const std::function<void(Asteroid&)>& fn) {
        for (auto& kv : sectors) for (auto& a : kv.second) fn(a);
    }
};

// ============================================================================
// SECTION 3: P2P swarm networking (UDP broadcast discovery + peer exchange
// + real-time snapshot sync), built on protocol.h / udp.h from this repo.
// ============================================================================

struct RemotePlayer {
    uint64_t id = 0;
    PeerAddr addr;
    std::string name;
    double lastSeen = 0;
    // Last two received snapshots for simple linear interpolation/extrapolation.
    Vec3d posA, posB;
    Quat  rotA, rotB;
    double tA = 0, tB = 0;
    ShipSnapshot last;
    bool everReceivedSnapshot = false;

    void feedSnapshot(const ShipSnapshot& s, double now) {
        posA = posB; rotA = rotB; tA = tB;
        posB = s.pos; rotB = s.rot; tB = now;
        last = s;
        everReceivedSnapshot = true;
    }
    // Render-time interpolated pose (falls back to latest known pose).
    void interpolated(double now, Vec3d& outPos, Quat& outRot) const {
        if (!everReceivedSnapshot) { outPos = posB; outRot = rotB; return; }
        double span = tB - tA;
        double t = span > 1e-6 ? clampf((float)((now - tA) / span), 0.f, 1.f) : 1.f;
        outPos = posA + (posB - posA) * (double)t;
        outRot = Quat::slerp(rotA, rotB, (float)t);
    }
};

class SwarmNetwork {
public:
    uint64_t localId;
    std::string localName;
    uint16_t port;
    UdpSocket sock;
    std::unordered_map<uint64_t, RemotePlayer> peers;
    uint32_t seqCounter = 0;

    double tLastDiscover = -999, tLastSnapshot = -999, tLastGossip = -999;
    static constexpr double DISCOVER_INTERVAL = 1.5;
    static constexpr double SNAPSHOT_INTERVAL = 0.05;  // 20 Hz
    static constexpr double GOSSIP_INTERVAL   = 2.0;
    static constexpr double PEER_TIMEOUT      = 6.0;

    int packetsIn = 0, packetsOut = 0;

    bool start(uint16_t desiredPort, const std::string& name) {
        localId = randomU64();
        localName = name;
        port = desiredPort;
        // If the default port is already taken by another local instance
        // (handy for testing several clients on one PC), probe upward.
        for (int attempt = 0; attempt < 8; attempt++) {
            if (sock.open(port, /*allowBroadcast=*/true, /*joinMulticast=*/true)) {
                logInfo("swarm: bound UDP port %u, peerId=%016llx name='%s'",
                        (unsigned)port, (unsigned long long)localId, name.c_str());
                return true;
            }
            port++;
        }
        logErr("swarm: could not bind any UDP port in range");
        return false;
    }

    // Optional direct bootstrap to a known internet peer (bypasses the need
    // for both sides to share a broadcast domain / LAN segment).
    void bootstrap(const std::string& ip, uint16_t remotePort) {
        PeerAddr a; a.ip = UdpSocket::ipFromString(ip); a.port = remotePort;
        if (a.ip == 0) { logErr("swarm: bad bootstrap address '%s'", ip.c_str()); return; }
        sendHello(a);
        logInfo("swarm: sent bootstrap HELLO to %s:%u", ip.c_str(), (unsigned)remotePort);
    }

    void sendHello(const PeerAddr& to) {
        ByteWriter w; writeHeader(w, PKT_HELLO);
        w.u64(localId); w.str(localName); w.u16(port);
        sock.sendToAddr(w.buf.data(), w.buf.size(), to);
        packetsOut++;
    }

    void sendWelcome(const PeerAddr& to) {
        ByteWriter w; writeHeader(w, PKT_WELCOME);
        w.u64(localId); w.str(localName);
        // Peer exchange payload: gossip everything we currently know.
        w.u16((uint16_t)std::min<size_t>(peers.size(), 32));
        int n = 0;
        for (auto& kv : peers) {
            if (n++ >= 32) break;
            w.u64(kv.second.id); w.u32(kv.second.addr.ip); w.u16(kv.second.addr.port); w.str(kv.second.name);
        }
        sock.sendToAddr(w.buf.data(), w.buf.size(), to);
        packetsOut++;
    }

    void sendPeerGossip(const PeerAddr& to) {
        ByteWriter w; writeHeader(w, PKT_PEERS);
        w.u64(localId);
        w.u16((uint16_t)std::min<size_t>(peers.size(), 32));
        int n = 0;
        for (auto& kv : peers) {
            if (n++ >= 32) break;
            w.u64(kv.second.id); w.u32(kv.second.addr.ip); w.u16(kv.second.addr.port); w.str(kv.second.name);
        }
        sock.sendToAddr(w.buf.data(), w.buf.size(), to);
        packetsOut++;
    }

    void sendSnapshot(const ShipState& ship, const Vec3d& absolutePos) {
        ByteWriter w; writeHeader(w, PKT_SNAPSHOT);
        ShipSnapshot s;
        s.peerId = localId;
        s.seq = seqCounter++;
        s.pos = absolutePos;
        s.vel = ship.vel;
        s.rot = ship.rot;
        s.throttle = (uint8_t)clampf(ship.throttle * 100.f, 0.f, 100.f);
        s.hull = (uint8_t)clampf(ship.hull, 0.f, 255.f);
        s.shield = (uint8_t)clampf(ship.shield, 0.f, 255.f);
        s.fuel = (uint8_t)clampf(ship.fuel, 0.f, 255.f);
        s.flags = ship.boosting ? SF_BOOST : 0;
        s.colorHash = (uint32_t)(localId & 0xFFFFFFFFu);
        ByteWriter body;
        body.u64(s.peerId); body.u32(s.seq); body.vec3d(s.pos); body.vec3(s.vel); body.quat(s.rot);
        body.u8(s.throttle); body.u8(s.shield); body.u8(s.armor); body.u8(s.hull);
        body.u8(s.fuel); body.u8(s.o2); body.u16(s.flags); body.f32(s.jumpCharge); body.u32(s.colorHash);
        w.raw(body.buf.data(), body.buf.size());
        for (auto& kv : peers) { sock.sendToAddr(w.buf.data(), w.buf.size(), kv.second.addr); packetsOut++; }
    }

    void sendBye() {
        ByteWriter w; writeHeader(w, PKT_BYE); w.u64(localId);
        for (auto& kv : peers) sock.sendToAddr(w.buf.data(), w.buf.size(), kv.second.addr);
    }

    void addOrTouchPeer(uint64_t id, const PeerAddr& addr, const std::string& name, double now) {
        if (id == localId) return;
        auto it = peers.find(id);
        if (it == peers.end()) {
            RemotePlayer rp; rp.id = id; rp.addr = addr; rp.name = name; rp.lastSeen = now;
            peers[id] = rp;
            logInfo("swarm: new peer '%s' (%016llx) from %s:%u — known peers=%zu",
                    name.c_str(), (unsigned long long)id, UdpSocket::ipToString(addr.ip).c_str(),
                    (unsigned)addr.port, peers.size());
            sendHello(addr); // ensure bidirectional handshake even if we learned about them via gossip
        } else {
            it->second.lastSeen = now;
            if (!name.empty()) it->second.name = name;
        }
    }

    void update(double now, const ShipState& ship, const Vec3d& absolutePos) {
        // 1) Broadcast discovery beacon.
        if (now - tLastDiscover > DISCOVER_INTERVAL) {
            tLastDiscover = now;
            ByteWriter w; writeHeader(w, PKT_DISCOVER);
            w.u64(localId); w.str(localName); w.u16(port);
            sock.broadcast(w.buf.data(), w.buf.size(), port);
            sock.multicast(w.buf.data(), w.buf.size(), port);
            packetsOut++;
        }
        // 2) 20Hz position/rotation/state sync to every known peer.
        if (now - tLastSnapshot > SNAPSHOT_INTERVAL) {
            tLastSnapshot = now;
            if (!peers.empty()) sendSnapshot(ship, absolutePos);
        }
        // 3) Decentralized peer-exchange gossip to a random known peer.
        if (now - tLastGossip > GOSSIP_INTERVAL && !peers.empty()) {
            tLastGossip = now;
            auto it = peers.begin();
            std::advance(it, rand() % peers.size());
            sendPeerGossip(it->second.addr);
        }
        // 4) Drain the socket.
        uint8_t buf[NET_MAX_PACKET];
        uint32_t fromIp; uint16_t fromPort;
        for (int guard = 0; guard < 256; guard++) {
            int n = sock.recvFrom(buf, sizeof(buf), fromIp, fromPort);
            if (n <= 0) break;
            packetsIn++;
            handlePacket(buf, (size_t)n, PeerAddr{ fromIp, fromPort }, now);
        }
        // 5) Peer timeout.
        for (auto it = peers.begin(); it != peers.end();) {
            if (now - it->second.lastSeen > PEER_TIMEOUT) {
                logInfo("swarm: peer '%s' timed out", it->second.name.c_str());
                it = peers.erase(it);
            } else ++it;
        }
    }

    void handlePacket(const uint8_t* data, size_t len, const PeerAddr& from, double now) {
        ByteReader r(data, len);
        PacketType type; uint8_t ver;
        if (!readHeader(r, type, ver)) return; // bad magic -> not our protocol, ignore
        if (ver != NET_VERSION) return;        // incompatible peer version

        switch (type) {
            case PKT_DISCOVER: {
                uint64_t id = r.u64(); std::string name = r.str(); uint16_t theirPort = r.u16();
                if (id == localId) break; // our own broadcast looped back
                PeerAddr real = from; real.port = theirPort ? theirPort : from.port;
                addOrTouchPeer(id, real, name, now);
                break;
            }
            case PKT_HELLO: {
                uint64_t id = r.u64(); std::string name = r.str(); uint16_t theirPort = r.u16();
                if (id == localId) break;
                PeerAddr real = from; real.port = theirPort ? theirPort : from.port;
                bool isNew = !peers.count(id);
                addOrTouchPeer(id, real, name, now);
                sendWelcome(real); // handshake completion + our peer list
                (void)isNew;
                break;
            }
            case PKT_WELCOME: {
                uint64_t id = r.u64(); std::string name = r.str();
                addOrTouchPeer(id, from, name, now);
                uint16_t count = r.u16();
                for (int i = 0; i < count && r.ok; i++) {
                    uint64_t pid = r.u64(); uint32_t pip = r.u32(); uint16_t pport = r.u16(); std::string pname = r.str();
                    if (pid != localId && !peers.count(pid) && pip != 0)
                        addOrTouchPeer(pid, PeerAddr{ pip, pport }, pname, now);
                }
                break;
            }
            case PKT_PEERS: {
                uint64_t id = r.u64();
                addOrTouchPeer(id, from, "", now);
                uint16_t count = r.u16();
                for (int i = 0; i < count && r.ok; i++) {
                    uint64_t pid = r.u64(); uint32_t pip = r.u32(); uint16_t pport = r.u16(); std::string pname = r.str();
                    if (pid != localId && !peers.count(pid) && pip != 0)
                        addOrTouchPeer(pid, PeerAddr{ pip, pport }, pname, now); // triggers HELLO -> full mesh
                }
                break;
            }
            case PKT_SNAPSHOT: {
                ShipSnapshot s;
                s.peerId = r.u64(); s.seq = r.u32(); s.pos = r.vec3d(); s.vel = r.vec3(); s.rot = r.quat();
                s.throttle = r.u8(); s.shield = r.u8(); s.armor = r.u8(); s.hull = r.u8();
                s.fuel = r.u8(); s.o2 = r.u8(); s.flags = r.u16(); s.jumpCharge = r.f32(); s.colorHash = r.u32();
                if (!r.ok || s.peerId == localId) break;
                auto it = peers.find(s.peerId);
                if (it == peers.end()) { addOrTouchPeer(s.peerId, from, "", now); it = peers.find(s.peerId); }
                if (it != peers.end()) { it->second.lastSeen = now; it->second.feedSnapshot(s, now); }
                break;
            }
            case PKT_BYE: {
                uint64_t id = r.u64();
                peers.erase(id);
                break;
            }
            default: break; // PKT_PING/PONG/CHAT/ENT_* intentionally unhandled in this demo
        }
    }

    void shutdown() { sendBye(); sock.close(); netPlatformShutdown(); }
};

// ============================================================================
// SECTION 4: bitmap-font HUD text (font8x8_basic -> textured quads)
// ============================================================================

struct BitmapFont {
    GLuint atlasTex = 0;
    static constexpr int GLYPHS = 128, GW = 8, GH = 8, COLS = 16, ROWS = 8;
    void build() {
        int atlasW = COLS * GW, atlasH = ROWS * GH;
        std::vector<unsigned char> rgba(atlasW * atlasH * 4, 0);
        for (int g = 0; g < GLYPHS; g++) {
            int gx = (g % COLS) * GW, gy = (g / COLS) * GH;
            for (int y = 0; y < GH; y++) {
                unsigned char row = font8x8_basic[g][y];
                for (int x = 0; x < GW; x++) {
                    bool on = (row >> x) & 1;
                    int px = gx + x, py = gy + y;
                    int idx = (py * atlasW + px) * 4;
                    rgba[idx + 0] = rgba[idx + 1] = rgba[idx + 2] = 255;
                    rgba[idx + 3] = on ? 255 : 0;
                }
            }
        }
        glGenTextures(1, &atlasTex);
        glBindTexture(GL_TEXTURE_2D, atlasTex);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, atlasW, atlasH, 0, GL_RGBA, GL_UNSIGNED_BYTE, rgba.data());
    }
    // Draws text in 2D screen space (call inside an ortho projection).
    void draw(float x, float y, float scale, Vec3 color, float alpha, const std::string& text) const {
        glEnable(GL_TEXTURE_2D);
        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
        glBindTexture(GL_TEXTURE_2D, atlasTex);
        glColor4f(color.x, color.y, color.z, alpha);
        glBegin(GL_QUADS);
        float cx = x;
        for (unsigned char c : text) {
            if (c >= GLYPHS) c = '?';
            int gx = (c % COLS), gy = (c / COLS);
            float u0 = gx / (float)COLS, u1 = (gx + 1) / (float)COLS;
            float v0 = gy / (float)ROWS, v1 = (gy + 1) / (float)ROWS;
            float gw = GW * scale, gh = GH * scale;
            glTexCoord2f(u0, v0); glVertex2f(cx, y);
            glTexCoord2f(u1, v0); glVertex2f(cx + gw, y);
            glTexCoord2f(u1, v1); glVertex2f(cx + gw, y + gh);
            glTexCoord2f(u0, v1); glVertex2f(cx, y + gh);
            cx += gw;
        }
        glEnd();
        glDisable(GL_BLEND);
        glDisable(GL_TEXTURE_2D);
    }
};

// ============================================================================
// SECTION 5: application
// ============================================================================

struct AppConfig {
    std::string assetDir = "assets";
    std::string playerName = "Pilot";
    uint16_t port = NET_PORT_DEFAULT;
    std::string bootstrapIp;
    uint16_t bootstrapPort = NET_PORT_DEFAULT;
    uint64_t universeSeed = 1337;
};

static void parseArgs(int argc, char** argv, AppConfig& cfg) {
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        if (a == "--assets" && i + 1 < argc) cfg.assetDir = argv[++i];
        else if (a == "--name" && i + 1 < argc) cfg.playerName = argv[++i];
        else if (a == "--port" && i + 1 < argc) cfg.port = (uint16_t)std::atoi(argv[++i]);
        else if (a == "--connect" && i + 1 < argc) {
            std::string v = argv[++i];
            auto pos = v.find(':');
            if (pos != std::string::npos) { cfg.bootstrapIp = v.substr(0, pos); cfg.bootstrapPort = (uint16_t)std::atoi(v.c_str() + pos + 1); }
            else cfg.bootstrapIp = v;
        }
        else if (a == "--seed" && i + 1 < argc) cfg.universeSeed = (uint64_t)std::strtoull(argv[++i], nullptr, 10);
    }
}

int main(int argc, char** argv) {
    AppConfig cfg;
    parseArgs(argc, argv, cfg);

    if (SDL_Init(SDL_INIT_VIDEO | SDL_INIT_TIMER) != 0) {
        logErr("SDL_Init failed: %s", SDL_GetError());
        return 1;
    }
    SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER, 1);
    SDL_GL_SetAttribute(SDL_GL_DEPTH_SIZE, 24);
    SDL_GL_SetAttribute(SDL_GL_RED_SIZE, 8);
    SDL_GL_SetAttribute(SDL_GL_GREEN_SIZE, 8);
    SDL_GL_SetAttribute(SDL_GL_BLUE_SIZE, 8);

    int winW = 1280, winH = 720;
    SDL_Window* window = SDL_CreateWindow("SpaceGame — P2P Deep Space",
        SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, winW, winH,
        SDL_WINDOW_OPENGL | SDL_WINDOW_RESIZABLE | SDL_WINDOW_SHOWN);
    if (!window) { logErr("SDL_CreateWindow failed: %s", SDL_GetError()); return 1; }

    SDL_GLContext glctx = SDL_GL_CreateContext(window);
    if (!glctx) { logErr("SDL_GL_CreateContext failed: %s", SDL_GetError()); return 1; }
    SDL_GL_SetSwapInterval(1); // vsync

    glEnable(GL_DEPTH_TEST);
    glEnable(GL_CULL_FACE);
    glEnable(GL_LIGHTING);
    glEnable(GL_LIGHT0);
    glEnable(GL_COLOR_MATERIAL);
    glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE);
    GLfloat sunDir[4] = { 0.4f, 0.6f, 0.5f, 0.0f }; // directional light
    glLightfv(GL_LIGHT0, GL_POSITION, sunDir);
    GLfloat ambient[4] = { 0.10f, 0.10f, 0.14f, 1.f };
    glLightModelfv(GL_LIGHT_MODEL_AMBIENT, ambient);
    glClearColor(0.01f, 0.01f, 0.03f, 1.f);

    // ---- Load assets -------------------------------------------------
    const std::string modelDir = cfg.assetDir + "/models/";
    const std::string texDir   = cfg.assetDir + "/textures/";

    std::vector<std::string> asteroidFiles = {
        "meteor.obj", "meteor_detailed.obj", "rock.obj",
        "rock_largeA.obj", "rock_largeB.obj",
        "rock_crystalsLargeA.obj", "rock_crystalsLargeB.obj"
    };
    std::vector<Mesh> asteroidMeshes;
    for (auto& f : asteroidFiles) {
        Mesh m = loadObjMesh(modelDir + f, modelDir);
        if (m.verts.empty()) m = makeFallbackCube(1.f, Vec3(0.5f, 0.45f, 0.4f));
        asteroidMeshes.push_back(std::move(m));
    }
    Mesh shipMesh = loadObjMesh(modelDir + "craft_speederA.obj", modelDir);
    if (shipMesh.verts.empty()) shipMesh = makeFallbackCube(1.f, Vec3(0.8f, 0.85f, 0.95f));
    Mesh remoteShipMesh = loadObjMesh(modelDir + "craft_speederC.obj", modelDir);
    if (remoteShipMesh.verts.empty()) remoteShipMesh = shipMesh;

    Texture texEarth, texMars, texSun;
    texEarth.load(texDir + "2k_earth.jpg");
    texMars.load(texDir + "2k_mars.jpg");
    texSun.load(texDir + "2k_sun.jpg");
    PlanetSphere sphereMesh = PlanetSphere::build(24, 36);

    std::vector<Planet> planets;
    planets.push_back({ "Sol",    Vec3d(0, 0, 0),          696.0, texSun });
    planets.push_back({ "Terra",  Vec3d(15000, 500, -4000), 120.0, texEarth });
    planets.push_back({ "Ares",   Vec3d(-9000, -200, 18000), 90.0, texMars });

    BitmapFont font; font.build();

    // Persistent starfield (fixed absolute world positions, drawn relative
    // to the floating origin exactly like everything else — no parallax
    // hacks needed because doubles keep the numbers exact).
    std::vector<Vec3d> stars;
    { Rng rng(0xC0FFEEu);
      for (int i = 0; i < 3000; i++) stars.push_back(rng.inUnitSphereD() * 200000.0); }

    // ---- World + player ------------------------------------------------
    SectorWorld world(cfg.universeSeed, (int)asteroidMeshes.size());
    ShipState ship;
    ship.pos = Vec3d(0, 0, 500);
    Vec3d origin(0, 0, 0);
    static constexpr double ORIGIN_SHIFT_THRESHOLD = 4000.0;

    // ---- Networking ------------------------------------------------
    SwarmNetwork net;
    if (!net.start(cfg.port, cfg.playerName)) logWarn("continuing without networking");
    if (!cfg.bootstrapIp.empty()) net.bootstrap(cfg.bootstrapIp, cfg.bootstrapPort);

    logInfo("=== SpaceGame P2P client online ===");
    logInfo("name='%s' port=%u seed=%llu", cfg.playerName.c_str(), (unsigned)net.port,
            (unsigned long long)cfg.universeSeed);
    logInfo("WASD pitch/yaw, Q/E roll, R/F throttle, SPACE boost, ESC quit");

    bool running = true;
    double lastTime = nowSeconds();
    double simTime = 0;

    while (running) {
        double t = nowSeconds();
        float dt = (float)std::min(0.05, t - lastTime); // clamp to avoid spiral-of-death on hitches
        lastTime = t;
        simTime += dt;

        // ---- Input --------------------------------------------------
        SDL_Event ev;
        while (SDL_PollEvent(&ev)) {
            if (ev.type == SDL_QUIT) running = false;
            else if (ev.type == SDL_KEYDOWN && ev.key.keysym.sym == SDLK_ESCAPE) running = false;
            else if (ev.type == SDL_WINDOWEVENT && ev.window.event == SDL_WINDOWEVENT_RESIZED) {
                winW = ev.window.data1; winH = ev.window.data2;
            }
        }
        const uint8_t* keys = SDL_GetKeyboardState(nullptr);
        Vec3 angVel(0, 0, 0);
        if (keys[SDL_SCANCODE_W]) angVel.x -= 1.f;
        if (keys[SDL_SCANCODE_S]) angVel.x += 1.f;
        if (keys[SDL_SCANCODE_A]) angVel.y += 1.f;
        if (keys[SDL_SCANCODE_D]) angVel.y -= 1.f;
        if (keys[SDL_SCANCODE_Q]) angVel.z += 1.f;
        if (keys[SDL_SCANCODE_E]) angVel.z -= 1.f;
        angVel *= 1.4f; // rad/s turn rate
        ship.rot = ship.rot.integrate(angVel, dt);

        if (keys[SDL_SCANCODE_R]) ship.throttle = clampf(ship.throttle + dt, 0.f, 1.f);
        if (keys[SDL_SCANCODE_F]) ship.throttle = clampf(ship.throttle - dt, 0.f, 1.f);
        ship.boosting = keys[SDL_SCANCODE_SPACE] && ship.fuel > 0.f;

        float thrustAccel = 60.f * ship.throttle * (ship.boosting ? 2.2f : 1.f);
        ship.vel += ship.rot.forward() * thrustAccel * dt;
        ship.vel *= (1.f - std::min(1.f, 0.15f * dt)); // gentle damping, not real-Newtonian on purpose
        if (ship.boosting) ship.fuel = std::max(0.f, ship.fuel - 12.f * dt);
        else ship.fuel = std::min(100.f, ship.fuel + 3.f * dt);
        ship.energy = std::min(100.f, ship.energy + (5.f - ship.throttle * 4.f) * dt);
        ship.energy = std::max(0.f, ship.energy);
        ship.shield = std::min(100.f, ship.shield + 2.f * dt);

        ship.pos += Vec3d(ship.vel) * (double)dt;

        // ---- Floating origin re-centering ---------------------------
        if ((ship.pos - origin).len() > ORIGIN_SHIFT_THRESHOLD) origin = ship.pos;

        // ---- World streaming -----------------------------------------
        world.streamAround(ship.pos);

        // ---- Simple asteroid collision (bounce off, chip hull) --------
        world.forEachAsteroid([&](Asteroid& a) {
            a.rot = a.rot.integrate(a.spin, dt);
            double d = (ship.pos - a.pos).len();
            double r = a.scale * 1.6;
            if (d < r) {
                Vec3d n = (ship.pos - a.pos).norm();
                ship.pos = a.pos + n * r;
                Vec3 nf = n.toFloat();
                ship.vel = ship.vel - nf * (2.f * ship.vel.dot(nf));
                ship.hull = std::max(0.f, ship.hull - 8.f * dt);
            }
        });

        // ---- Networking update ----------------------------------------
        net.update(t, ship, ship.pos);

        // ================= RENDER =================
        glViewport(0, 0, winW, winH);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        float aspect = winH > 0 ? (float)winW / (float)winH : 1.f;
        Mat4 proj = Mat4::perspective(deg2rad(70.f), aspect, 0.1f, 300000.f);
        glMatrixMode(GL_PROJECTION);
        glLoadMatrixf(proj.m);

        // Third-person chase camera, positioned/oriented purely from the
        // ship's rotation; camera "position" is always the floating origin
        // reference point (0 in relative-space), which is exactly why the
        // floating-origin trick eliminates jitter at huge coordinates.
        Vec3 camOffset = ship.rot.forward() * -18.f + ship.rot.up() * 5.f;
        Vec3 camFwd = ((ship.pos.relativeTo(origin)) - (camOffset + ship.pos.relativeTo(origin))).norm(); // placeholder, replaced below
        Vec3 shipRel = ship.pos.relativeTo(origin);
        Vec3 camPosRel = shipRel + camOffset;
        Vec3 lookDir = (shipRel - camPosRel).norm();
        Mat4 view = Mat4::viewFromBasis(lookDir, ship.rot.up(), ship.rot.right());
        Mat4 camTranslate = Mat4::translate(camPosRel * -1.f);
        // view = rotation-only; translate camera to origin-of-view manually via glMultMatrix order
        glMatrixMode(GL_MODELVIEW);
        glLoadMatrixf(view.m);
        glMultMatrixf(camTranslate.m);

        GLfloat lightPos[4] = { 0.4f, 0.7f, 0.5f, 0.f };
        glLightfv(GL_LIGHT0, GL_POSITION, lightPos);

        // ---- Starfield --------------------------------------------------
        glDisable(GL_LIGHTING);
        glPointSize(1.6f);
        glColor3f(0.85f, 0.9f, 1.0f);
        glBegin(GL_POINTS);
        for (auto& s : stars) { Vec3 rp = s.relativeTo(origin); glVertex3f(rp.x, rp.y, rp.z); }
        glEnd();
        glEnable(GL_LIGHTING);

        // ---- Planets ------------------------------------------------
        double nearestDist = 1e18; std::string nearestName;
        for (auto& p : planets) {
            Vec3 rp = p.pos.relativeTo(origin);
            glPushMatrix();
            glTranslatef(rp.x, rp.y, rp.z);
            sphereMesh.draw(p.radius, p.tex.id);
            glPopMatrix();
            double d = (ship.pos - p.pos).len() - p.radius;
            if (d < nearestDist) { nearestDist = d; nearestName = p.name; }
        }

        // ---- Asteroids ------------------------------------------------
        world.forEachAsteroid([&](Asteroid& a) {
            Vec3 rp = a.pos.relativeTo(origin);
            double d = (ship.pos - a.pos).len() - a.scale;
            if (d < nearestDist) { nearestDist = d; nearestName = "asteroid"; }
            // simple frustum-independent distance cull to keep the demo light
            if (rp.len() > 6000.f) return;
            Mat4 m = Mat4::TRS(rp, a.rot, Vec3(a.scale));
            glPushMatrix();
            glMultMatrixf(m.m);
            asteroidMeshes[a.meshIdx].draw();
            glPopMatrix();
        });

        // ---- Local ship (visible in third-person) -----------------------
        {
            Mat4 m = Mat4::TRS(shipRel, ship.rot, Vec3(3.0f));
            glPushMatrix();
            glMultMatrixf(m.m);
            shipMesh.draw();
            glPopMatrix();
        }

        // ---- Remote players --------------------------------------------
        for (auto& kv : net.peers) {
            RemotePlayer& rpv = kv.second;
            Vec3d ipos; Quat irot;
            rpv.interpolated(t, ipos, irot);
            Vec3 rp = ipos.relativeTo(origin);
            double d = (ship.pos - ipos).len();
            if (d < nearestDist) { nearestDist = d; nearestName = rpv.name.empty() ? "peer" : rpv.name; }
            Mat4 m = Mat4::TRS(rp, irot, Vec3(3.0f));
            glPushMatrix();
            glMultMatrixf(m.m);
            remoteShipMesh.draw();
            glPopMatrix();
        }

        // ================= HUD (orthographic overlay) =================
        glDisable(GL_DEPTH_TEST);
        glDisable(GL_LIGHTING);
        glMatrixMode(GL_PROJECTION);
        glPushMatrix();
        glLoadIdentity();
        glOrtho(0, winW, winH, 0, -1, 1);
        glMatrixMode(GL_MODELVIEW);
        glPushMatrix();
        glLoadIdentity();

        auto drawBar = [&](float x, float y, float w, float h, float frac, Vec3 col) {
            glColor3f(0.08f, 0.08f, 0.10f);
            glBegin(GL_QUADS);
            glVertex2f(x, y); glVertex2f(x + w, y); glVertex2f(x + w, y + h); glVertex2f(x, y + h);
            glEnd();
            glColor3f(col.x, col.y, col.z);
            float fw = w * clampf(frac, 0.f, 1.f);
            glBegin(GL_QUADS);
            glVertex2f(x, y); glVertex2f(x + fw, y); glVertex2f(x + fw, y + h); glVertex2f(x, y + h);
            glEnd();
        };

        float hx = 24, hy = winH - 110.f;
        drawBar(hx, hy,        220, 16, ship.hull   / 100.f, Vec3(0.85f, 0.2f, 0.2f));
        drawBar(hx, hy + 22,   220, 16, ship.shield  / 100.f, Vec3(0.2f, 0.55f, 0.95f));
        drawBar(hx, hy + 44,   220, 16, ship.energy  / 100.f, Vec3(0.95f, 0.85f, 0.2f));
        drawBar(hx, hy + 66,   220, 16, ship.fuel    / 100.f, Vec3(0.3f, 0.9f, 0.4f));
        font.draw(hx + 226, hy,      1.2f, Vec3(1,1,1), 1.f, "HULL");
        font.draw(hx + 226, hy + 22, 1.2f, Vec3(1,1,1), 1.f, "SHLD");
        font.draw(hx + 226, hy + 44, 1.2f, Vec3(1,1,1), 1.f, "ENRG");
        font.draw(hx + 226, hy + 66, 1.2f, Vec3(1,1,1), 1.f, "FUEL");

        char distBuf[32]; formatDistance(nearestDist, distBuf, sizeof(distBuf));
        std::ostringstream rangeLine;
        rangeLine << "RANGE " << distBuf << " -> " << (nearestName.empty() ? "---" : nearestName);
        font.draw(24, 20, 1.6f, Vec3(0.6f, 1.f, 0.8f), 1.f, rangeLine.str());

        std::ostringstream posLine;
        posLine << "POS " << (long long)ship.pos.x << "," << (long long)ship.pos.y << "," << (long long)ship.pos.z
                << "  THR " << (int)(ship.throttle * 100) << "%"
                << (ship.boosting ? "  BOOST" : "");
        font.draw(24, 44, 1.4f, Vec3(0.8f, 0.85f, 1.f), 1.f, posLine.str());

        std::ostringstream netLine;
        netLine << "SWARM peers=" << net.peers.size() << " port=" << net.port
                << " in=" << net.packetsIn << " out=" << net.packetsOut;
        font.draw(24, 66, 1.4f, Vec3(1.f, 0.75f, 0.3f), 1.f, netLine.str());

        // Peer bearing radar (top-right compass of known ships).
        float radarCx = winW - 110.f, radarCy = 110.f, radarR = 80.f;
        glColor4f(0.1f, 1.f, 0.6f, 0.15f);
        glBegin(GL_TRIANGLE_FAN);
        glVertex2f(radarCx, radarCy);
        for (int i = 0; i <= 32; i++) { float a = i / 32.f * TAU; glVertex2f(radarCx + std::cos(a) * radarR, radarCy + std::sin(a) * radarR); }
        glEnd();
        glColor3f(0.5f, 1.f, 0.7f);
        glPointSize(6.f);
        glBegin(GL_POINTS);
        for (auto& kv : net.peers) {
            Vec3d ipos; Quat irot; kv.second.interpolated(t, ipos, irot);
            Vec3 toPeer = (ipos - ship.pos).toFloat();
            float fwdDot = toPeer.norm().dot(ship.rot.forward());
            float rightDot = toPeer.norm().dot(ship.rot.right());
            glVertex2f(radarCx + rightDot * radarR * 0.85f, radarCy - fwdDot * radarR * 0.85f);
        }
        glEnd();
        font.draw(radarCx - 30, radarCy + radarR + 8, 1.2f, Vec3(0.6f, 1.f, 0.8f), 1.f, "SWARM RADAR");

        glMatrixMode(GL_PROJECTION); glPopMatrix();
        glMatrixMode(GL_MODELVIEW); glPopMatrix();
        glEnable(GL_DEPTH_TEST);
        glEnable(GL_LIGHTING);

        SDL_GL_SwapWindow(window);
    }

    net.shutdown();
    SDL_GL_DeleteContext(glctx);
    SDL_DestroyWindow(window);
    SDL_Quit();
    return 0;
}
