#include <iostream>
#include <fstream>
#include <cmath>
#include <vector>

#define TINYOBJLOADER_IMPLEMENTATION
#include "objloader.hpp"

#define WIDTH 800
#define HEIGHT 800

using namespace std;

const double epsilon = 1e-9; // Small value

struct vec3
{
    vec3() : x(0), y(0), z(0) {}
    vec3(double v) : x(v), y(v), z(v) {}
    vec3(double x0, double y0, double z0 = 0) : x(x0), y(y0), z(z0) {}
    vec3 operator*(double a) const { return vec3(x * a, y * a, z * a); }
    vec3 operator*(const vec3 r) const { return vec3(x * r.x, y * r.y, z * r.z); }
    vec3 operator/(const double r) const { return fabs(r) > epsilon ? vec3(x / r, y / r, z / r) : vec3(0, 0, 0); }
    vec3 operator+(const vec3 &v) const { return vec3(x + v.x, y + v.y, z + v.z); }
    vec3 operator-(const vec3 &v) const { return vec3(x - v.x, y - v.y, z - v.z); }
    vec3 operator-() const { return vec3(-x, -y, -z); }
    void operator+=(const vec3 &v) { x += v.x, y += v.y, z += v.z; }
    void operator*=(double a) { x *= a, y *= a, z *= a; }
    void operator*=(const vec3 &v) { x *= v.x, y *= v.y, z *= v.z; }
    double length() const { return sqrt(x * x + y * y + z * z); }
    double average() { return (x + y + z) / 3; }
    vec3 normalized() const { return (*this) / length(); }

    double x, y, z;
};

double
dot(const vec3 &v1, const vec3 &v2)
{
    return (v1.x * v2.x + v1.y * v2.y + v1.z * v2.z);
}

vec3 cross(const vec3 &v1, const vec3 &v2)
{
    return vec3(v1.y * v2.z - v1.z * v2.y, v1.z * v2.x - v1.x * v2.z, v1.x * v2.y - v1.y * v2.x);
}

ostream &operator<<(ostream &os, vec3 vec)
{
    os << vec.x << " " << vec.y << " " << vec.z;
    return os;
}

template <typename T>
inline T clamp(T val, T low, T high)
{
    return max(min(val, high), low);
}

struct Ray
{
    Ray(vec3 o, vec3 d)
        : origin(o), dir(d){};
    vec3 origin, dir;
    double t;
};

struct Triangle {
    Triangle(const vec3 a_, const vec3 b_, const vec3 c_)
    : a(a_), b(b_), c(c_) {
        //TODO compute normals

    };

    Triangle(const vec3 a_, const vec3 b_, const vec3 c_,
             const vec3 na_, const vec3 nb_, const vec3 nc_)
    : a(a_), b(b_), c(c_), na(na_), nb(nb_), nc(nc_) {};

    vec3 a, b, c;
    vec3 na, nb, nc;
};

/*struct Sphere { 
    Sphere(double rad_, vec3 p_): 
        rad(rad_), p(p_) {} 
   
    double intersect(const Ray &r) const { // returns distance, 0 if nohit 
        vec3 op = p - r.origin; // Solve t^2*d.d + 2*t*(o-p).d + (o-p).(o-p)-R^2 = 0 
        double t, eps=1e-4, b=dot(op, r.dir), det=b*b-dot(op,op)+rad*rad; 
        
        if (det<0) 
          return INFINITY; 
        else det=sqrt(det); 

        return (t=b-det)>eps ? t : ((t=b+det)>eps ? t : INFINITY); 
    } 

    double rad;  // radius 
    vec3 p;     // position
}; 

const Sphere * findNearest(Ray & r, const vector<Sphere> & scene) {

  const Sphere * nearest = nullptr;
  double d = INFINITY, di;

  for (size_t i = 0; i < scene.size(); i++)
  {
    di = scene[i].intersect(r);

    if (di != INFINITY && di < d)
    {
      nearest = &scene[i];
      d = di;
    }
  }
  
  r.t = d;
  return nearest;
}*/

struct Camera
{
    Camera(vec3 pos_, vec3 up_, vec3 front_, double fov_)
        : pos(pos_), up(up_), front(front_), fov(fov_)
    {
        side = cross(front, up).normalized();

        double gw = 2.0 * tan(fov / 2.0);
        double gh = gw * HEIGHT / WIDTH;

        d0 = front - (side * gw / 2) + (up * gh / 2);
        m = side * (gw / (WIDTH - 1));
        n = -up * (gh / (HEIGHT - 1));
    };

    //function generating ray for given pixel coorinates
    Ray primaryRayForPixel(double x, double y)
    {
        vec3 dir = (d0 + m * x + n * y).normalized();
        return Ray(pos, dir);
    }

    vec3 pos, up, front, side, m, n, d0;
    double fov;
};

class Scene
{
public:
    Scene(const char *path, const char *file)
    {
        std::string inputfile = file;
        tinyobj::ObjReaderConfig reader_config;
        reader_config.mtl_search_path = path;

        tinyobj::ObjReader reader;

        if (!reader.ParseFromFile(inputfile, reader_config))
        {
            if (!reader.Error().empty())
                std::cerr << "TinyObjReader: " << reader.Error();
            exit(1);
        }

        if (!reader.Warning().empty())
            std::cout << "TinyObjReader: " << reader.Warning();

        auto &attrib = reader.GetAttrib();
        auto &shapes = reader.GetShapes();
        //auto &materials = reader.GetMaterials();
        bool normals_loaded = attrib.normals.size() != 0;
        vec3 v[3], n[3];

        // Loop over shapes
        for (size_t s = 0; s < shapes.size(); s++)
        {
            // Loop over faces(polygon)
            size_t index_offset = 0;
            for (size_t f = 0; f < shapes[s].mesh.num_face_vertices.size(); f++)
            {
                int fv = shapes[s].mesh.num_face_vertices[f];

                if (fv != 3)
                {
                    cerr << "Encountered face with more than 3 vertices" << endl;
                    abort();
                }

                // Loop over vertices in the face.
                for (int i = 0; i < fv; i++)
                {
                    // access to vertex
                    tinyobj::index_t idx = shapes[s].mesh.indices[index_offset + i];
                    tinyobj::real_t vx = attrib.vertices[3 * idx.vertex_index + 0];
                    tinyobj::real_t vy = attrib.vertices[3 * idx.vertex_index + 1];
                    tinyobj::real_t vz = attrib.vertices[3 * idx.vertex_index + 2];
                    v[i] = vec3(vx, vy, vz);

                    if (normals_loaded)
                    {
                        // access to normal
                        tinyobj::real_t nx = attrib.normals[3 * idx.normal_index + 0];
                        tinyobj::real_t ny = attrib.normals[3 * idx.normal_index + 1];
                        tinyobj::real_t nz = attrib.normals[3 * idx.normal_index + 2];
                        n[i] = vec3(nx, ny, nz);
                    }
                }
                index_offset += fv;

                // per-face material
                //shapes[s].mesh.material_ids[f];
                if (normals_loaded)
                    triangles.push_back(new Triangle(v[0], v[1], v[2], n[0], n[1], n[2]));
                else
                    triangles.push_back(new Triangle(v[0], v[1], v[2]));
            }
        }
    }

    ~Scene()
    {
        for (size_t i = 0; i < triangles.size(); ++i)
        {
            delete triangles[i];
        }
    }

private:
    vector<Triangle *> triangles;
};

void saveImage(float *image, size_t width, size_t height)
{
    //image saving
    ofstream output("output.ppm");
    output << "P3\n"
           << WIDTH << " " << HEIGHT << "\n"
           << 255 << endl;
    for (size_t i = 0; i < width * height * 3; ++i)
        output << (int)image[i] << " ";
    output.close();
}

int main(int argc, char const *argv[])
{
    float *image = new float[WIDTH * HEIGHT * 3];

    //creating checker texture
    /*for(int i = 0; i < WIDTH; ++i)
        for(int j = 0; j < HEIGHT; ++j)
            for(int k = 0; k < 3; ++k)
                image[3 * (j * WIDTH + i) + k] = (i + j) % 2 * 255;*/

    Camera camera(vec3(0), vec3(0, 0, 1), vec3(0, 1, 0), 1.0);

    /*vector<Sphere> scene;
    scene.push_back(Sphere(2, vec3(0, 10, 0)));
    scene.push_back(Sphere(3.5, vec3(-5, 15, 0)));
    scene.push_back(Sphere(5, vec3(5, 20, 0)));
    scene.push_back(Sphere(10e5, vec3(0, 0, -10e5 - 5)));*/

    for (int j = 0; j < HEIGHT; ++j)
        for (int i = 0; i < WIDTH; ++i)
        {
            Ray r = camera.primaryRayForPixel(i, j);

            //vrhni paprsek co sceny
            //otestuj prusek s kazdou sferou
            //vezmi nejblizsi pokud existuje
            /*const Sphere * sphere = findNearest(r, scene);

            //do obrazku zapis hloubku
            if (sphere != nullptr) {
            for(int k = 0; k < 3; ++k)
                image[(j * WIDTH + i) * 3 + k] = 255;//clamp(r.t, 0.0, 255.0);
            }*/
        }

    //image saving
    //saveImage(image, WIDTH, HEIGHT);
    delete [] image;
}
