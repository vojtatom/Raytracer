#include <iostream>
#include <fstream>
#include <cmath>
#include <vector>
#include <random>

#define TINYOBJLOADER_IMPLEMENTATION
#include "objloader.hpp"

#define WIDTH 400
#define HEIGHT 400
#define LIGHTSAMPLES 32

using namespace std;

//const double epsilon = 1e-5; // Small value
const double epsilon = 1e-1; // Small value

struct vec3
{
    vec3() : x(0), y(0), z(0) {}
    vec3(double v) : x(v), y(v), z(v) {}
    vec3(const tinyobj::real_t v[3]) : x(v[0]), y(v[1]), z(v[2]) {}
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
    void operator/=(const int i) { x /= i, y /= i, z /= i; }
    double operator[](const size_t i) {return ((double*)this)[i]; }
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
        : origin(o), dir(d), b1(0), b2(0) {};
    vec3 origin, dir;
    float b1, b2;
    double t;
};

struct Material {
    vec3 Kd; //diffuse
    vec3 Ks; //specular
    vec3 Ke; //emission
    vec3 T; //transmittance
    double ior; //index of refraction
    double h; //shinines

    vec3 phongSimple(const vec3 & x, const vec3 & lightPosition, const vec3 & viewerPosition, const vec3 & normal) 
    {
        vec3 lightDirection = (lightPosition - x).normalized();
        vec3 viewerDirection = (viewerPosition - x).normalized();
        double lightNormalDot = clamp(dot(lightDirection, normal), 0.0, 1.0);
        vec3 idealReflectedDirection =  normal * 2 * lightNormalDot - lightDirection;
       
        vec3 diffuse = Kd * lightNormalDot;
        vec3 specular = Ks * pow(clamp(dot(viewerDirection, idealReflectedDirection), 0.0, 1.0), h);

        return diffuse + specular; 
    }
};

struct Triangle {
    Triangle(const vec3 a_, const vec3 b_, const vec3 c_, Material * mat_)
    : a(a_), b(b_), c(c_), mat(mat_) 
    {
        //TODO compute normals
        vec3 n = cross(b - a, c - a);
        na = nb = nc = n.normalized();
        edgeAB = b - a;
        edgeAC = c - a;
    };

    Triangle(const vec3 a_, const vec3 b_, const vec3 c_,
             const vec3 na_, const vec3 nb_, const vec3 nc_, Material * mat_)
    : a(a_), b(b_), c(c_), na(na_), nb(nb_), nc(nc_), mat(mat_) 
    {
        edgeAB = b - a;
        edgeAC = c - a;
    };


    float intersect(const Ray & ray, bool inside, float & b1, float & b2)
    {
        vec3 e1(b - a), e2(c - a);
        vec3 pvec = cross(ray.dir, e2);
        float det = dot(e1, pvec);

        if (inside)
        {
            //do not cull backface
            if (fabs(det) < epsilon) // ray is parallel to triangle
                return INFINITY;
        }
        else
        {
            //cull backface
            if (det < epsilon) // ray is parallel to triangle
                return INFINITY;
        }

        float invDet = 1.0f / det;

        // Compute first barycentric coordinate
        vec3 tvec = ray.origin - a;
        b1 = dot(tvec, pvec) * invDet;

        if (b1 < 0.0f || b1 > 1.0f)
            return INFINITY;

        // Compute second barycentric coordinate
        vec3 qvec = cross(tvec, e1);
        b2 = dot(ray.dir, qvec) * invDet;

        if (b2 < 0.0f || b1 + b2 > 1.0f)
            return INFINITY;

        // Compute t to intersection point
        float t = dot(e2, qvec) * invDet;
        return t;
    }

    vec3 normal(const float b1, const float b2) const
    {
        return (na * (1 - b1 - b2) + nb * b1 + nc * b2).normalized();
    }

    void samplePoint(vec3 & sampleP, vec3 & sampleN) const {
        float u = ((float) rand()) / RAND_MAX;
        float v = ((float) rand()) / RAND_MAX;

        if (u + v > 1) {
            u = 1 - u;
            v = 1 - v;
        }

        sampleP = a + edgeAB * u + edgeAC * v;
        sampleN = normal(u, v);
    }

    double emissionWeight(const vec3 & lightPoint, const vec3 & lightNormal, const vec3 & x) 
    {
        double area = 0.5 * cross(edgeAB, edgeAC).length();
        vec3 shadowRayDirection = (x - lightPoint);
        double lightDistance = shadowRayDirection.length();

        return (area * clamp(dot(lightNormal, shadowRayDirection.normalized()), 0.0 , 1.0)) / (LIGHTSAMPLES * lightDistance * lightDistance); 
    }


    vec3 a, b, c;
    vec3 na, nb, nc;
    vec3 edgeAB, edgeAC;
    Material * mat;
};

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
    Scene(const char *pathToScenes, const char *pathToOBJ)
    {
        std::string inputfile = pathToOBJ;
        tinyobj::ObjReaderConfig reader_config;
        reader_config.mtl_search_path = pathToScenes;

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
        auto &materials_ = reader.GetMaterials();

        for(size_t i = 0; i < materials_.size(); ++i) 
        {
            materials.push_back(new Material{vec3(materials_[i].diffuse),
                                             vec3(materials_[i].specular),
                                             vec3(materials_[i].emission),
                                             vec3(materials_[i].transmittance),
                                             1 / materials_[i].ior,
                                             materials_[i].shininess});
        }


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
                Material * mat = materials[shapes[s].mesh.material_ids[f]];
                if (normals_loaded)
                    triangles.push_back(new Triangle(v[0], v[1], v[2], n[0], n[1], n[2], mat));
                else
                    triangles.push_back(new Triangle(v[0], v[1], v[2], mat));

                if (triangles.back()->mat->Ke.average() > 0.0)
                    lights.push_back(triangles.back());
            }
        }
    };

    Triangle * findNearest(Ray & ray, bool inside) {
        Triangle * nearest = nullptr;
        ray.t = INFINITY;
        double t = INFINITY;
        float b1, b2;

        for (size_t i = 0; i < triangles.size(); i++)
        {
            t = triangles[i]->intersect(ray, inside, b1, b2);

            if ((t != INFINITY && t < ray.t) && t > epsilon)
            {
                nearest = triangles[i];
                ray.t = t;
                ray.b1 = b1;
                ray.b2 = b2;
            }
        }

        return nearest;
    };

    ~Scene()
    {
        for (size_t i = 0; i < triangles.size(); ++i)
        {
            delete triangles[i];
        }

        for (size_t i = 0; i < materials.size(); ++i)
        {
            delete materials[i];
        }
    };

    vec3 calculatePointLightLocation() {
        vec3 location;
        for (size_t i = 0; i < lights.size(); i++)
        {
            location += lights[i]->a;
            location += lights[i]->b;
            location += lights[i]->c;
        }
        
        location /= lights.size() * 3;
        return location;
    }

    bool ocluded(const vec3 & source, const vec3 & dest) {
        vec3 dir = dest - source;
        double dist = (dir).length();

        Ray ray(source, dir.normalized());
        Triangle * tri = findNearest(ray, false);

        if (!tri)
            return false;

        if(ray.t < (dist - epsilon))
            return true;
        return false;

    }

    vec3 trace(Ray ray, bool inside) { 
        vec3 color(0.0);
        const Triangle * tri = findNearest(ray, inside);
        
        //prusecik
        const vec3 x = ray.origin + ray.dir * ray.t;

        if (!tri)
            return color;

        vec3 normal = tri->normal(ray.b1, ray.b2); 
        vec3 lightPosition, lightNormal;

        for (size_t i = 0; i < lights.size(); i++)
        {
            for (size_t j = 0; j < LIGHTSAMPLES; j++)
            {
                lights[i]->samplePoint(lightPosition, lightNormal);
                if(!ocluded(x, lightPosition))
                {
                    color += lights[i]->mat->Ke
                             * tri->mat->phongSimple(x, lightPosition, ray.origin, normal)
                             * lights[i]->emissionWeight(lightPosition, lightNormal, x);
                }
            }
        }

        /*float lightFactor = 1.0 / (lights.size());
        color *= lightFactor;*/

        //odrazene paprsky
        if (tri->mat->Ks.average() > 0) {
            //spoleham na to ze ray.dir je normalizovany
            vec3 idealReflectedDirection =  normal * 2 * dot(-ray.dir, normal) + ray.dir;
            Ray reflected(x, idealReflectedDirection.normalized());
            vec3 reflectedColor = trace(reflected, inside);
            color += reflectedColor * tri->mat->Ks;
        }

        //lomene paprsky
        if (tri->mat->T.average() > 0) {

            double ior = tri->mat->ior;
            
            if (inside) {
                ior = 1.0 / ior;
                normal = -normal;
            }

            vec3 view = -ray.dir;
            double vnDot = dot(view, normal);

            double termSqrt = 1 - (ior * ior) * (1 - vnDot * vnDot);
            
            if (termSqrt > 0) {
                double normalFactor =  ior * vnDot - sqrt(termSqrt);
                vec3 refractedDirection = view * (-ior) + normal * normalFactor;
                Ray refracted(x, refractedDirection.normalized());
                vec3 refractedColor = trace(refracted, !inside);
                color += refractedColor * tri->mat->T;
            } else {
                //TODO total reflection
            }
        }

        color += tri->mat->Ke;
        return color;

    };



private:
    vector<Triangle *> triangles;
    vector<Triangle *> lights;
    vector<Material *> materials;
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


    Camera camera(vec3(278, 273, -1000), vec3(0, 1, 0), vec3(0, 0, 1), 0.6);
    //Camera camera(vec3(0, 1, 4.42), vec3(0, 1, 0), vec3(0, 0, -1), 0.6);
    Scene scene("./scenes", "./scenes/CornellBox-Original.obj");
    //Scene scene("./scenes", "./scenes/CornellBox-Sphere.obj");

    double maxt = 0;

    for (int j = 0; j < HEIGHT; ++j)
        for (int i = 0; i < WIDTH; ++i)
        {
            Ray r = camera.primaryRayForPixel(i, j);
            vec3 color = scene.trace(r, false);
            cout << "\rrendering " << (double)((j * WIDTH) + i) * 100 / (WIDTH * HEIGHT) << "%" << flush;

            for(int k = 0; k < 3; ++k)
                image[(j * WIDTH + i) * 3 + k] = clamp(color[k] * 255, 0.0, 255.0);
            
        }

    //image saving
    saveImage(image, WIDTH, HEIGHT);
    delete [] image;
}
