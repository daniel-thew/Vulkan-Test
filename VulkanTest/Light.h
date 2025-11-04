#pragma once
class Light
{
public:
	// If point light, direction[3] = 1
	// If directional, direction[3] = 0
	glm::vec4 position;
	float fov;
	Light(glm::vec4 p, float f) {
		position = p;
		fov = f;
	}
};