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
	vec4 lightPos;
} ubo;

// Vertex attributes
layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inColor;
layout(location = 2) in vec2 inTexCoord;
layout(location = 3) in vec3 inNormal;

layout(location = 0) out vec3 fragColor;
layout(location = 1) out vec2 fragTexCoord;
layout(location = 2) out vec3 fragNormal;
layout(location = 3) out vec3 fragPos;
layout(location = 4) out vec4 fragLightPos;


void main(){
	gl_Position = ubo.proj * ubo.view * ubo.model * vec4(inPosition, 1.0);
	fragPos = vec3(ubo.view * ubo.model * vec4(inPosition, 1.0));
	fragNormal = mat3(transpose(inverse(ubo.view * ubo.model))) * inNormal;
	fragLightPos = ubo.view * ubo.lightPos;

	fragColor = inColor;
	fragTexCoord = inTexCoord;
}