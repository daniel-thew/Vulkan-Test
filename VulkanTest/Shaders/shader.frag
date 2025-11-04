#version 450

layout(location = 0) in vec3 fragColor;
layout(location = 1) in vec2 fragTexCoord;
layout(location = 2) in vec3 fragNormal;
layout(location = 3) in vec3 fragPos;
layout(location = 4) in vec4 fragLightPos;

layout(binding = 1) uniform sampler2D texSampler;

layout(location = 0) out vec4 outColor;

void main(){
	float ambientStrength = 0.3;
	// light color is just vec3(1.0f) for now
	vec3 texColor = texture(texSampler, fragTexCoord).rgb;
	vec3 ambient = vec3(ambientStrength);

	vec3 normal = normalize(fragNormal);
	vec3 lightDirection;
	if(fragLightPos.w == 1){
		lightDirection = normalize(vec3(fragLightPos) - fragPos);
	}
	else{
		vec4 tempLightDirection = normalize(-fragLightPos);
		lightDirection = vec3(tempLightDirection);
	}
	float diff = max(dot(normal, lightDirection), 0.0);
	vec3 diffuse = diff * vec3(1.0);

	vec3 result = (ambient + diffuse) * texColor * fragColor;

	outColor = vec4(result, 1.0);
	//outColor = vec4(normalize(fragNormal) * 0.5 + 0.5, 1.0);
	//outColor = texture(texSampler, fragTexCoord);
}