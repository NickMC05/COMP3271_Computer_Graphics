#version 330 core
out vec4 FragColor;

in vec3 FragPos;
in vec3 Normal;
in vec2 TexCoords;

uniform vec3 lightPos;
uniform vec3 viewPos;
uniform vec3 objectColor;

void main() {

    float ambientStrength = 0.1;
    float specularStrength = 0.3;
    float diffuseStrength = 1.0;
    float shininess = 0.8;
    float lightStrength = 0.5;
    float ambientLightStrength = 1.0;

    float ambient = 1;
    float diffuse = 1;
    float specular = 1;


    // TODO: Implement Blinn-Phong/Phong lighting model
    // Some useful functions:
    // - normalize(vec3 v): returns a normalized vector
    // - dot(vec3 a, vec3 b): returns the dot product of two vectors
    // - pow(float x, float y): returns x raised to the power of y
    // - max(float x, float y): returns the maximum value between x and y
    // - reflect(vec3 v, vec3 n): returns the reflection direction of the incident vector v and the normal n

    ambient = ambientLightStrength * ambientStrength;
    diffuse = max(lightStrength * diffuseStrength * dot(Normal, lightPos), 0);
    specular = lightStrength * specularStrength * pow(max(dot(reflect(-lightPos, Normal), viewPos), 0), shininess);

    vec3 result = (ambient + diffuse + specular) * objectColor;
    FragColor = vec4(result, 1.0);
}