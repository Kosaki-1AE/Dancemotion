#include <assimp/Importer.hpp>
#include <assimp/postprocess.h>
#include <assimp/scene.h>
#include <GLFW/glfw3.h>

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>
#include <algorithm>

struct Vec3 { float x{}, y{}, z{}; };
struct Tri { Vec3 a, b, c, n; };

struct MeshData {
    std::vector<Tri> tris;
    Vec3 min{0,0,0}, max{0,0,0}, center{0,0,0};
    float radius{1.0f};
};

static Vec3 sub(Vec3 a, Vec3 b){ return {a.x-b.x, a.y-b.y, a.z-b.z}; }
static Vec3 cross(Vec3 a, Vec3 b){ return {a.y*b.z-a.z*b.y, a.z*b.x-a.x*b.z, a.x*b.y-a.y*b.x}; }
static Vec3 norm(Vec3 v){
    float l = std::sqrt(v.x*v.x + v.y*v.y + v.z*v.z);
    if(l <= 1e-8f) return {0,0,1};
    return {v.x/l, v.y/l, v.z/l};
}

static void add_node_meshes(const aiScene* scene, aiNode* node, MeshData& out){
    for(unsigned i=0; i<node->mNumMeshes; ++i){
        aiMesh* m = scene->mMeshes[node->mMeshes[i]];
        for(unsigned f=0; f<m->mNumFaces; ++f){
            const aiFace& face = m->mFaces[f];
            if(face.mNumIndices < 3) continue;
            Vec3 v0{m->mVertices[face.mIndices[0]].x, m->mVertices[face.mIndices[0]].y, m->mVertices[face.mIndices[0]].z};
            for(unsigned k=1; k+1<face.mNumIndices; ++k){
                Vec3 v1{m->mVertices[face.mIndices[k]].x, m->mVertices[face.mIndices[k]].y, m->mVertices[face.mIndices[k]].z};
                Vec3 v2{m->mVertices[face.mIndices[k+1]].x, m->mVertices[face.mIndices[k+1]].y, m->mVertices[face.mIndices[k+1]].z};
                Vec3 n = norm(cross(sub(v1,v0), sub(v2,v0)));
                out.tris.push_back({v0,v1,v2,n});
            }
        }
    }
    for(unsigned i=0; i<node->mNumChildren; ++i) add_node_meshes(scene, node->mChildren[i], out);
}

static bool load_model(const std::string& path, MeshData& out){
    Assimp::Importer importer;
    const aiScene* scene = importer.ReadFile(path,
        aiProcess_Triangulate | aiProcess_JoinIdenticalVertices | aiProcess_GenNormals | aiProcess_SortByPType);
    if(!scene || !scene->mRootNode){
        std::cerr << "load error: " << importer.GetErrorString() << "\n";
        return false;
    }
    add_node_meshes(scene, scene->mRootNode, out);
    if(out.tris.empty()){
        std::cerr << "no triangles found\n";
        return false;
    }

    out.min = out.max = out.tris[0].a;
    auto scan = [&](Vec3 v){
        out.min.x = std::min(out.min.x, v.x); out.min.y = std::min(out.min.y, v.y); out.min.z = std::min(out.min.z, v.z);
        out.max.x = std::max(out.max.x, v.x); out.max.y = std::max(out.max.y, v.y); out.max.z = std::max(out.max.z, v.z);
    };
    for(const auto& t: out.tris){ scan(t.a); scan(t.b); scan(t.c); }
    out.center = {(out.min.x+out.max.x)/2, (out.min.y+out.max.y)/2, (out.min.z+out.max.z)/2};
    float dx=out.max.x-out.min.x, dy=out.max.y-out.min.y, dz=out.max.z-out.min.z;
    out.radius = std::max({dx,dy,dz,1.0f});
    return true;
}

static float rotX=20.0f, rotY=-35.0f, zoom=2.5f;
static bool dragging=false; static double lastX=0, lastY=0;

static void cursor_cb(GLFWwindow* w, double x, double y){
    if(dragging){ rotY += float(x-lastX)*0.4f; rotX += float(y-lastY)*0.4f; }
    lastX=x; lastY=y; (void)w;
}
static void mouse_cb(GLFWwindow* w, int button, int action, int mods){
    if(button == GLFW_MOUSE_BUTTON_LEFT) dragging = (action == GLFW_PRESS);
    (void)w; (void)mods;
}
static void scroll_cb(GLFWwindow* w, double xoff, double yoff){
    zoom *= (yoff > 0) ? 0.90f : 1.10f;
    zoom = std::clamp(zoom, 0.2f, 50.0f);
    (void)w; (void)xoff;
}
static void key_cb(GLFWwindow* w, int key, int scancode, int action, int mods){
    if(action == GLFW_PRESS && (key == GLFW_KEY_ESCAPE || key == GLFW_KEY_Q)) glfwSetWindowShouldClose(w, GLFW_TRUE);
    (void)scancode; (void)mods;
}

static void draw_axes(float s){
    glBegin(GL_LINES);
    glColor3f(1,0,0); glVertex3f(0,0,0); glVertex3f(s,0,0);
    glColor3f(0,1,0); glVertex3f(0,0,0); glVertex3f(0,s,0);
    glColor3f(0,0.6f,1); glVertex3f(0,0,0); glVertex3f(0,0,s);
    glEnd();
}

static void draw_mesh(const MeshData& m){
    glBegin(GL_TRIANGLES);
    for(const auto& t: m.tris){
        glNormal3f(t.n.x,t.n.y,t.n.z);
        glColor3f(0.78f,0.78f,0.82f);
        glVertex3f(t.a.x,t.a.y,t.a.z);
        glVertex3f(t.b.x,t.b.y,t.b.z);
        glVertex3f(t.c.x,t.c.y,t.c.z);
    }
    glEnd();
}

int main(int argc, char** argv){
    if(argc < 2){
        std::cerr << "usage: kos3d <model.stl|model.3mf|model.obj|model.glb>\n";
        std::cerr << "note: .f3d/.f3z are Fusion native files; export them to STL/3MF first.\n";
        return 1;
    }
    MeshData mesh;
    if(!load_model(argv[1], mesh)) return 2;
    std::cout << "loaded: " << argv[1] << "\ntriangles: " << mesh.tris.size() << "\n";
    std::cout << "size: " << (mesh.max.x-mesh.min.x) << " x " << (mesh.max.y-mesh.min.y) << " x " << (mesh.max.z-mesh.min.z) << "\n";

    if(!glfwInit()){ std::cerr << "glfw init failed\n"; return 3; }
    GLFWwindow* win = glfwCreateWindow(1000, 750, "kos3d - Linux command 3D viewer", nullptr, nullptr);
    if(!win){ glfwTerminate(); return 4; }
    glfwMakeContextCurrent(win);
    glfwSetCursorPosCallback(win, cursor_cb);
    glfwSetMouseButtonCallback(win, mouse_cb);
    glfwSetScrollCallback(win, scroll_cb);
    glfwSetKeyCallback(win, key_cb);
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_LIGHTING); glEnable(GL_LIGHT0);
    float light[] = {1.5f, 2.0f, 3.0f, 0.0f};
    glLightfv(GL_LIGHT0, GL_POSITION, light);

    while(!glfwWindowShouldClose(win)){
        int w,h; glfwGetFramebufferSize(win,&w,&h);
        glViewport(0,0,w,h);
        glClearColor(0.08f,0.09f,0.10f,1);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glMatrixMode(GL_PROJECTION); glLoadIdentity();
        float aspect = h? float(w)/float(h):1.0f;
        float nearp=0.01f, farp=1000.0f, fov=45.0f;
        float top = nearp * std::tan(fov*3.1415926f/360.0f);
        glFrustum(-top*aspect, top*aspect, -top, top, nearp, farp);
        glMatrixMode(GL_MODELVIEW); glLoadIdentity();
        glTranslatef(0,0,-mesh.radius*zoom);
        glRotatef(rotX,1,0,0); glRotatef(rotY,0,1,0);
        glScalef(2.0f/mesh.radius, 2.0f/mesh.radius, 2.0f/mesh.radius);
        glTranslatef(-mesh.center.x, -mesh.center.y, -mesh.center.z);
        draw_axes(mesh.radius*0.6f);
        draw_mesh(mesh);
        glfwSwapBuffers(win);
        glfwPollEvents();
    }
    glfwDestroyWindow(win); glfwTerminate();
    return 0;
}
