#include <iostream>
#include <fstream>
#include <cmath>
#include <vector>

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


struct Ray {
  Ray(vec3 o, vec3 d)
  : origin(o), direction(d) {};
  vec3 origin, direction;
  double t;
};


struct Triangle {
  vec3 x, y, z;
};


struct Sphere { 
    Sphere(double rad_, vec3 p_): 
        rad(rad_), p(p_) {} 
   
    double intersect(const Ray &r) const { // returns distance, 0 if nohit 
        vec3 op = p - r.origin; // Solve t^2*d.d + 2*t*(o-p).d + (o-p).(o-p)-R^2 = 0 
        double t, eps=1e-4, b=dot(op, r.direction), det=b*b-dot(op,op)+rad*rad; 
        
        if (det<0) 
          return INFINITY; 
        else det=sqrt(det); 

        return (t=b-det)>eps ? t : ((t=b+det)>eps ? t : INFINITY); 
    } 

    double rad;  // radius 
    vec3 p;     // position
}; 


struct Camera {
  Camera(vec3 pos_, vec3 up_, vec3 front_, double fov_)
  : pos(pos_), up(up_), front(front_), fov(fov_) {
    side = cross(front, up).normalized();

    double gw = 2.0 * tan(fov / 2.0);
    double gh = gw * HEIGHT / WIDTH;

    d0 = front - (side * gw / 2) + (up * gh / 2);
    m = side * (gw / (WIDTH - 1));
    n = -up * (gh / (HEIGHT - 1));

  };

  //function generating ray for given pixel coorinates
  Ray primaryRayForPixel(double x, double y) {
    vec3 dir = (d0 + m * x + n * y).normalized();
    return Ray(pos, dir);
  }

  vec3 pos, up, front, side, m, n, d0;
  double fov;
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
}


int main(int argc, char const *argv[])
{
    float * image = new float[WIDTH * HEIGHT * 3];

    //creating checker texture
    /*for(int i = 0; i < WIDTH; ++i)
        for(int j = 0; j < HEIGHT; ++j)
            for(int k = 0; k < 3; ++k)
                image[3 * (j * WIDTH + i) + k] = (i + j) % 2 * 255;*/

    Camera camera(vec3(0), vec3(0, 0, 1), vec3(0, 1, 0), 1.0);

    vector<Sphere> scene;
    scene.push_back(Sphere(2, vec3(0, 10, 0)));
    scene.push_back(Sphere(3.5, vec3(-5, 15, 0)));
    scene.push_back(Sphere(5, vec3(5, 20, 0)));
    scene.push_back(Sphere(10e5, vec3(0, 0, -10e5 - 5)));


    for(int j = 0; j < HEIGHT; ++j)
      for(int i = 0; i < WIDTH; ++i)
      {
        Ray r = camera.primaryRayForPixel(i, j);

        //vrhni paprsek co sceny
        //otestuj prusek s kazdou sferou
        //vezmi nejblizsi pokud existuje
        const Sphere * sphere = findNearest(r, scene);

        //do obrazku zapis hloubku
        if (sphere != nullptr) {
          for(int k = 0; k < 3; ++k)
            image[(j * WIDTH + i) * 3 + k] = 255 - clamp(r.t * 10, 0.0, 255.0);
        }
      }


    //image saving
    ofstream output("output.ppm");   
    output << "P3\n" << WIDTH << " " << HEIGHT << "\n" << 255 << endl;
    for (int i = 0; i < WIDTH * HEIGHT * 3; ++i) 
        output << (int) image[i] << " ";
    output.close();
}
