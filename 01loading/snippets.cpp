
/*
// Möller-Trumbore algorithm
// Find intersection point - from PBRT - www.pbrt.org
float intersect(Ray & ray, bool cullback,  float & b1, float & b2)
{
    vec3 e1(b - a), e2(c - a);
    vec3 pvec = cross(ray.dir, e2);
    float det = dot(e1, pvec);

    if (cullback)
    {
        if (det < epsilon) // ray is parallel to triangle
            return INFINITY;
    }
    else
    {
        if (fabs(det) < epsilon) // ray is parallel to triangle
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
*/