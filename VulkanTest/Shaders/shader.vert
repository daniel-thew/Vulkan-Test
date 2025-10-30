#version 450

// Alignment info:
// scalars have to be aligned by N
// vec2 aligned by 2N
// vec3 or vec4 aligned by 4N
// nested structure must be aligned by the base alignment of its members
// rounded up to a multiple of 16
// mat4 must have the same alignment as a vec4
// can use this function: alignas(16)

// UBO stuff
layout(binding = 0) uniform UniformBufferObject{
	mat4 model;
	mat4 view;
	mat4 proj;
} ubo;

// Vertex attributes
layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inColor;
layout(location = 2) in vec2 inTexCoord;

layout(location = 0) out vec3 fragColor;
layout(location = 1) out vec2 fragTexCoord;


void main(){
	gl_Position = ubo.proj * ubo.view * ubo.model * vec4(inPosition, 1.0);
	fragColor = inColor;
	fragTexCoord = inTexCoord;
}