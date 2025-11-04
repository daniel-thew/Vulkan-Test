#pragma once

// Provides functions, structures, and enums
// Using the #define allows GLFW to include its own definitions
// and automatically load the Vulkan header with it
#define GLFW_INCLUDE_VULKAN
// Make sure our rotations are based in radians
#define GLM_FORCE_RADIANS
// Make the perspective projection matrix use 0.0 - 1.0 range for depth
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
// Alignment for shaders
#define GLM_FORCE_DEFAULT_ALIGNED_GENTYPES
// For image loading
#define STB_IMAGE_WRITE_IMPLEMENTATION
#define STB_IMAGE_IMPLEMENTATION
// For model loading
#define TINYGLTF_IMPLEMENTATION
// For hashing vertices
#define GLM_ENABLE_EXPERIMENTAL

#include <GLFW/glfw3.h>
// GLM for linear algebra stuff
#include <glm/glm.hpp>
// Included for reporting and propagating errors
#include <iostream>
#include <stdexcept>
// Provides EXIT_SUCCESS and EXIT_FAILURE macros
#include <cstdlib>
// Vector library
#include <vector>
// Used for determining if a value exists or not
#include <optional>
// For unique queue families
#include <set>
// For numeric limits
#include <limits>
// For clamp
#include <algorithm>
// For reading shader files (SPIR V)
#include <fstream>
// For vertex buffer stuff
#include <array>
// For matrix transformations
#include <glm/gtc/matrix_transform.hpp>
// For timekeeping
#include <chrono>
// For image loading
//#include <stb_image.h>
// For model loading
#include <tiny_gltf.h>
// For keeping track of unique vertices to deduplicate vertices
#include <unordered_map>
// For vertex hashing for the unordered map
#include <glm/gtx/hash.hpp>

// For GameObjects
#include "GameObject.h"
// Camera
#include "Camera.h"
// Light
#include "Light.h"
