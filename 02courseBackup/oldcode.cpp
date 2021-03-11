//unused old snippets of code from previous exercises

/*
struct Sphere
{
    Sphere(double rad_, vec3 p_) : rad(rad_), p(p_) {}

    double intersect(const Ray &r) const
    {							// returns distance, 0 if nohit
        vec3 op = p - r.origin; // Solve t^2*d.d + 2*t*(o-p).d + (o-p).(o-p)-R^2 = 0
        double t, eps = 1e-4, b = dot(op, r.direction), det = b * b - dot(op, op) + rad * rad;

        if (det < 0)
            return INFINITY;
        else
            det = sqrt(det);

        return (t = b - det) > eps ? t : ((t = b + det) > eps ? t : INFINITY);
    }

    double rad; // radius
    vec3 p;		// position
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

const Sphere *findNearest(Ray &r, const vector<Sphere> &scene)
{

    const Sphere *nearest = nullptr;
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