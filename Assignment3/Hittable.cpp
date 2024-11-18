#include "Hittable.h"


// Sphere
bool Sphere::Hit(const Ray &ray, HitRecord *hit_record) const {
    // TODO 1: Implement the intersection test between a ray and a sphere.
    // You should return true if the ray intersects the sphere, and false otherwise.
    // you could use the following member variables:
    // o_: center of sphere
    // r_: radius
    // material_: data structure storing the shading parameters
    // You should also initialize the hit_record with the intersection information.
    // hit_record->position: the intersection position
    // hit_record->normal: the normal vector at the intersection position
    // hit_record->distance: the distance from the ray's origin to the intersection position
    // hit_record->in_direction: the direction of the shoot in ray
    // hit_record->reflection: the direction of the reflected ray
    // hit_record->material: the material of the sphere

    // Variables
    Point center_of_sphere = o_;
    float radius = r_;
    Material material = material_;

    Point ray_origin = ray.o;
    Vec ray_distance = ray.d;

    // Find delta
    glm::vec3 center_distance = ray_origin - center_of_sphere;
    float a = glm::dot(ray_distance, ray_distance);
    float b = 2 * glm::dot(ray_distance, center_distance);
    float c = glm::dot(center_distance, center_distance) - (radius * radius);
    float delta = b * b - 4.f * a * c;

    // Check distance
    float distance = 0.0f;
    if (delta == 0.0f)
    {
        distance = -b / (2 * a);
    }
    else
    {
        float distance1 = (-b + sqrt(delta)) / (2 * a);
        float distance2 = (-b - sqrt(delta)) / (2 * a);

        if (distance1 < distance2) distance = distance1;
        else distance = distance2;
    }

    // If no intersection then return
    if (distance <= 0) return false;

    // Initialize the hit_record with the intersection information
    hit_record->position = ray_origin + distance * ray_distance;
    hit_record->normal = glm::normalize(hit_record->position - center_of_sphere);
    hit_record->distance = distance;
    hit_record->in_direction = ray_distance;
    hit_record->reflection = glm::normalize(ray_distance - 2 * glm::dot(ray_distance, hit_record->normal) * hit_record->normal);
    hit_record->material = material;

    return true;
}

// Quadric
bool Quadric::Hit(const Ray &ray, HitRecord *hit_record) const {

    glm::vec4 D(ray.d, 0.f);
    glm::vec4 S(ray.o, 1.f);

    float a = glm::dot(D, A_ * D);
    float b = 2.f * glm::dot(S, A_ * D);
    float c = glm::dot(S, A_ * S);
    float delta = b * b - 4.f * a * c;

    if (delta < 1e-5f) {
        return false;
    }

    float t = glm::min((-b - glm::sqrt(delta)) / (2.f * a), (-b + glm::sqrt(delta)) / (2.f * a));
    if (t < 1e-3f) {
        return false;
    }

    Point hit_pos = ray.o + ray.d * t;
    Vec normal = glm::normalize(glm::vec3((A_ + glm::transpose(A_)) * glm::vec4(hit_pos, 1.f)));
    hit_record->position = hit_pos;
    hit_record->normal = normal;
    hit_record->distance = t;
    hit_record->in_direction = ray.d;
    hit_record->reflection = ray.d - 2.f * glm::dot(ray.d, hit_record->normal) * hit_record->normal;
    hit_record->material = material_;
    return true;

}

// Triangle
bool Triangle::Hit(const Ray &ray, HitRecord *hit_record) const {
    // TODO 2: Implement the intersection test between a ray and a triangle.
    // You should return true if the ray intersects the triangle, and false otherwise.
    // you could use the following member variables:
    // a_, b_, c_: vertices of triangle
    // n_a_, n_b_, n_c_: corresponding vertex normal vectors
    // phong_interpolation_: flag of using Phong shading or flat shading

    // You should also initialize the hit_record with the intersection information.
    // hit_record->position: the intersection position
    // hit_record->normal: the normal vector at the intersection position
    // hit_record->distance: the distance from the ray's origin to the intersection position
    // hit_record->in_direction: the direction of the shoot in ray
    // hit_record->reflection: the direction of the reflected ray

    // Hint: possible functions you may use
    // glm::normalize
    // glm::cross
    // glm::dot
    // glm::length

    // Variables
    Point vertice1 = a_;
    Point vertice2 = b_;
    Point vertice3 = c_;

    Vec normal_vertice1 = n_a_;
    Vec normal_vertice2 = n_b_;
    Vec normal_vertice3 = n_c_;

    bool phong_interpolation = phong_interpolation_;

    Point ray_origin = ray.o;
    Vec ray_distance = ray.d;

    // Find normal
    glm::vec3 normal = glm::cross(vertice2 - vertice1, vertice3 - vertice1);

    // If invalid then no intersection
    if (glm::dot(ray_distance, normal) == 0.0f) return false;
    
    // Find distance
    float distance = (glm::dot(normal, vertice1) - glm::dot(normal, ray_origin)) / glm::dot(normal, ray_distance);

    // If distance is near 0 then no intersection
    if (distance <= 1e-5f) return false;

    // Find directions
    glm::vec3 position = ray_origin + distance * ray_distance;

    glm::vec3 origin_vertice1 = position - vertice1;
    glm::vec3 origin_vertice2 = position - vertice2;
    glm::vec3 origin_vertice3 = position - vertice3;

    glm::vec3 cross_origin_1_2 = glm::cross(origin_vertice1, origin_vertice2);
    glm::vec3 cross_origin_2_3 = glm::cross(origin_vertice2, origin_vertice3);
    glm::vec3 cross_origin_3_1 = glm::cross(origin_vertice3, origin_vertice1);

    // If direction not same then return false
    if (glm::dot(cross_origin_1_2, cross_origin_2_3) <= 0 || glm::dot(cross_origin_2_3, cross_origin_3_1) <= 0 || glm::dot(cross_origin_3_1, cross_origin_1_2) <= 0) return false;

    // Find final normal
    if (phong_interpolation)
    {
        // If phong
        float length_origin_1_2 = glm::length(cross_origin_1_2) / 2;
        float length_origin_2_3 = glm::length(cross_origin_2_3) / 2;
        float length_origin_3_1 = glm::length(cross_origin_3_1) / 2;
        float area = glm::length(normal) / 2;

        float alphaA = length_origin_2_3 / area;
        float alphaB = length_origin_3_1 / area;
        float alphaC = length_origin_1_2 / area;

        normal = alphaA * normal_vertice1 + alphaB * normal_vertice2 + alphaC * normal_vertice3;
    }
    else
    {
        // If no phong
        normal = glm::normalize(normal);
    }

    // Initialize the hit_record with the intersection information
    hit_record->position = position;
    hit_record->normal = normal;
    hit_record->distance = distance;
    hit_record->in_direction = ray_distance;
    hit_record->reflection = glm::normalize(ray_distance - 2 * glm::dot(ray_distance, normal) * normal);

    return true;
}

// ---------------------------------------------------------------------------------------------
// ------------------------------ no need to change --------------------------------------------
// ---------------------------------------------------------------------------------------------

// CompleteTriangle
bool CompleteTriangle::Hit(const Ray &ray, HitRecord *hit_record) const {
    bool ret = triangle_.Hit(ray, hit_record);
    if (ret) {
        hit_record->material = material_;
    }
    return ret;
}


// Mesh
Mesh::Mesh(const std::string &file_path,
           const Material &material,
           bool phong_interpolation) :
        ply_data_(file_path), material_(material), phong_interpolation_(phong_interpolation) {
    std::vector<std::array<double, 3>> v_pos = ply_data_.getVertexPositions();
    vertices_.resize(v_pos.size());

    for (int i = 0; i < vertices_.size(); i++) {
        vertices_[i] = Point(v_pos[i][0], v_pos[i][1], v_pos[i][2]);
    }

    f_ind_ = ply_data_.getFaceIndices();

    // Calc face normals
    for (const auto &face: f_ind_) {
        Vec normal = glm::normalize(
                glm::cross(vertices_[face[1]] - vertices_[face[0]], vertices_[face[2]] - vertices_[face[0]]));
        face_normals_.emplace_back(normal);
    }

    // Calc vertex normals
    vertex_normals_.resize(vertices_.size(), Vec(0.f, 0.f, 0.f));
    for (int i = 0; i < f_ind_.size(); i++) {
        for (int j = 0; j < 3; j++) {
            vertex_normals_[f_ind_[i][j]] += face_normals_[i];
        }
    }
    for (auto &vertex_normal: vertex_normals_) {
        vertex_normal = glm::normalize(vertex_normal);
    }

    // Construct hittable triangles
    for (const auto &face: f_ind_) {
        triangles_.emplace_back(vertices_[face[0]], vertices_[face[1]], vertices_[face[2]],
                                vertex_normals_[face[0]], vertex_normals_[face[1]], vertex_normals_[face[2]],
                                phong_interpolation_);
    }

    // Calc bounding box
    Point bbox_min(1e5f, 1e5f, 1e5f);
    Point bbox_max(-1e5f, -1e5f, -1e5f);
    for (const auto &vertex: vertices_) {
        bbox_min = glm::min(bbox_min, vertex - 1e-3f);
        bbox_max = glm::max(bbox_max, vertex + 1e-3f);
    }

    // Build Octree
    tree_nodes_.emplace_back(new OctreeNode());
    tree_nodes_.front()->bbox_min = bbox_min;
    tree_nodes_.front()->bbox_max = bbox_max;

    root_ = tree_nodes_.front().get();
    for (int i = 0; i < f_ind_.size(); i++) {
        InsertFace(root_, i);
    }

    area = 0.0f;
    for (auto &triangle: triangles_) {
        area += triangle.getArea();
    }
    emission = material_.emission;
}

bool Mesh::Hit(const Ray &ray, HitRecord *hit_record) const {
    const bool brute_force = false;
    if (brute_force) {
        // Naive hit algorithm
        float min_dist = 1e5f;
        for (const auto &triangle: triangles_) {
            HitRecord curr_hit_record;
            if (triangle.Hit(ray, &curr_hit_record)) {
                if (curr_hit_record.distance < min_dist) {
                    *hit_record = curr_hit_record;
                    min_dist = curr_hit_record.distance;
                }
            }
        }
        if (min_dist + 1.0 < 1e5f) {
            hit_record->material = material_;
            return true;
        }
        return false;
    } else {
        bool ret = OctreeHit(root_, ray, hit_record);
        if (ret) {
            hit_record->material = material_;
        }
        return ret;
    }
}

bool Mesh::IsFaceInsideBox(const std::vector<size_t> &face, const Point &bbox_min, const Point &bbox_max) const {
    for (size_t idx: face) {
        const auto &pt = vertices_[idx];
        for (int i = 0; i < 3; i++) {
            if (pt[i] < bbox_min[i] + 1e-6f) return false;
            if (pt[i] > bbox_max[i] - 1e-6f) return false;
        }
    }
    return true;
}

bool Mesh::IsRayIntersectBox(const Ray &ray, const Point &bbox_min, const Point &bbox_max) const {
    float t_min = -1e5f;
    float t_max = 1e5f;

    for (int i = 0; i < 3; i++) {
        if (glm::abs(ray.d[i]) < 1e-6f) {
            if (ray.o[i] < bbox_min[i] + 1e-6f || ray.o[i] > bbox_max[i] - 1e-6f) {
                t_min = 1e5f;
                t_max = -1e5f;
            }
        } else {
            if (ray.d[i] > 0.f) {
                t_min = glm::max(t_min, (bbox_min[i] - ray.o[i]) / ray.d[i]);
                t_max = glm::min(t_max, (bbox_max[i] - ray.o[i]) / ray.d[i]);
            } else {
                t_min = glm::max(t_min, (bbox_max[i] - ray.o[i]) / ray.d[i]);
                t_max = glm::min(t_max, (bbox_min[i] - ray.o[i]) / ray.d[i]);
            }
        }
    }

    return t_min + 1e-6f < t_max;
}

void Mesh::InsertFace(OctreeNode *u, size_t face_idx) {
    const Point &bbox_min = u->bbox_min;
    const Point &bbox_max = u->bbox_max;

    Vec bias = bbox_max - bbox_min;
    Vec half_bias = bias * 0.5f;

    bool inside_childs = false;

    for (size_t a = 0; a < 2; a++) {
        for (size_t b = 0; b < 2; b++) {
            for (size_t c = 0; c < 2; c++) {
                size_t child_idx = ((a << 2) | (b << 1) | c);
                Point curr_bbox_min = bbox_min + half_bias * Vec(float(a), float(b), float(c));
                Point curr_bbox_max = curr_bbox_min + half_bias;
                if (IsFaceInsideBox(f_ind_[face_idx], curr_bbox_min, curr_bbox_max)) {
                    if (u->childs[child_idx] == nullptr) {
                        tree_nodes_.emplace_back(new OctreeNode());
                        OctreeNode *child = tree_nodes_.back().get();
                        u->childs[child_idx] = tree_nodes_.back().get();
                        child->bbox_min = curr_bbox_min;
                        child->bbox_max = curr_bbox_max;
                    }
                    InsertFace(u->childs[child_idx], face_idx);
                    inside_childs = true;
                }
            }
        }
    }

    if (!inside_childs) {
        u->face_index.push_back(face_idx);
    }
}

bool Mesh::OctreeHit(OctreeNode *u, const Ray &ray, HitRecord *hit_record) const {
    if (!IsRayIntersectBox(ray, u->bbox_min, u->bbox_max)) {
        return false;
    }
    float distance = 1e5f;
    for (const auto &face_idx: u->face_index) {
        HitRecord curr_hit_record;
        if (triangles_[face_idx].Hit(ray, &curr_hit_record)) {
            if (curr_hit_record.distance < distance) {
                distance = curr_hit_record.distance;
                *hit_record = curr_hit_record;
            }
        }
    }

    for (const auto &child: u->childs) {
        if (child == nullptr) {
            continue;
        }
        HitRecord curr_hit_record;
        if (OctreeHit(child, ray, &curr_hit_record)) {
            if (curr_hit_record.distance < distance) {
                distance = curr_hit_record.distance;
                *hit_record = curr_hit_record;
            }
        }
    }
    return distance + 1 < 1e5f;
}


// Hittable list
void HittableList::PushHittable(const Hittable &hittable) {
    hittable_list_.push_back(&hittable);
}

bool HittableList::Hit(const Ray &ray, HitRecord *hit_record) const {
    float min_dist = 1e5f;
    for (const auto &hittable: hittable_list_) {
        HitRecord curr_hit_record;
        if (hittable->Hit(ray, &curr_hit_record)) {
            if (curr_hit_record.distance < min_dist) {
                *hit_record = curr_hit_record;
                min_dist = curr_hit_record.distance;
            }
        }
    }
    return min_dist + 1.0 < 1e4f;
}
