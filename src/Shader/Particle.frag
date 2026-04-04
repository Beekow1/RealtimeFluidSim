#version 330 core

out vec4 FragColor;
in vec3 vColor;

void main() {
    vec2 coord = gl_PointCoord * 2.0 - 1.0;
    float dist = dot(coord, coord);

    if (dist > 1.0) {
        discard;
    }

    FragColor = vec4(vColor, 1.0);
}