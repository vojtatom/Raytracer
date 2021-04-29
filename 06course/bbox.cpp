struct BBox{
    BBox() : low(RTINFINITY), high(-RTINFINITY), id(boxid++) {};

    void reset()
    {
        low = vec3(RTINFINITY);
        high = vec3(-RTINFINITY);
    }

    void include(Triangle * tri)
    {
        for(size_t i = 0; i < 3; i++)
        {
            low[i] =  min(low[i], tri->a[i]);
            low[i] =  min(low[i], tri->b[i]);
            low[i] =  min(low[i], tri->c[i]);
            high[i] =  max(high[i], tri->a[i]);
            high[i] =  max(high[i], tri->b[i]);
            high[i] =  max(high[i], tri->c[i]);
        }
    }

    void include(BBox & box)
    {
        for(size_t i = 0; i < 3; i++)
        {
            low[i] =  min(low[i], box.low[i]);
            high[i] =  max(high[i], box.high[i]);
        }
    }

    float midpoint(unsigned char axis) const {
        return (low[axis] + high[axis]) * 0.5;
    }

    bool intersects(Ray & ray, float & min_t, float & max_t) const {
        float tmin, tmax;

        vec3 invDir = 1.f / ray.dir;

        tmin = (low.x - ray.origin.x) * invDir.x;
        tmax = (high.x - ray.origin.x) * invDir.x;

        if (tmin > tmax)
            swap(tmin, tmax);

        float tymin, tymax;

        tymin = (low.y - ray.origin.y) * invDir.y;
        tymax = (high.y - ray.origin.y) * invDir.y;

        if (tymin > tymax)
            swap(tymin, tymax);

        if ((tmax < tymin) || (tmin > tymax))
            return false;
        

        if (tymax < tmax)
            tmax = tymax;
        
        if (tmin < tymin)
            tmin = tymin;

        float tzmin, tzmax;

        tzmin = (low.z - ray.origin.z) * invDir.z;
        tzmax = (high.z - ray.origin.z) * invDir.z;
         

        if (tzmin > tzmax)
            swap(tzmin, tzmax);

        if ((tmax < tzmin) || (tmin > tzmax))
            return false;

        if (tzmax < tmax)
            tmax = tzmax;
        
        if (tmin < tzmin)
            tmin = tzmin;

        min_t = tmin;
        max_t = tmax;
        return true;
    }

    vec3 low;
    vec3 high;
    size_t id;
};