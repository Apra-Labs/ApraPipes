
#include <stdlib.h>
#include <iostream>
#include <stddef.h>
#include <math.h>
//yash cahnge
// #include <GL/gl.h>
#ifndef GL_H
#define GL_H
#include <GL/glew.h>
#include <GL/glut.h>
#endif
// #include <opencv2/opencv.hpp>
#include "Matrix.h"
#include "Program.h"
#include "GLUtils.h"

struct point {
	float x;
	float y;
	float z;
} __attribute__((packed));

struct color {
	float r;
	float g;
	float b;
} __attribute__((packed));

// Each vertex has position, normal and color:
struct vertex {
	struct point pos;
	struct point normal;
	struct color color;
} __attribute__((packed));

// Each triangle has three vertices:
struct triangle {
	struct vertex vert[3];
} __attribute__((packed));

// Each corner point has a position and a color:
struct corner {
	struct point pos;
	struct color color;
} __attribute__((packed));

// Each face has a single normal, four corner points,
// and two triangles:
struct face {
	struct corner corner[4];
	struct point normal;
	struct triangle tri[2];
} __attribute__((packed));

// Each cube has six faces:
struct cube {
	struct face face[6];
} __attribute__((packed));

static GLuint vao, vbo;
static float matrix[16] = { 0 };

// Mouse movement:
static struct {
	int x;
	int y;
} pan;

// Cube rotation axis:
static struct point rot = {
	.x = 0.0f,
	.y = 1.0f,
	.z = 0.0f,
};

// Return the cross product of two vectors:
static void
cross (struct point *result, const struct point *a, const struct point *b)
{
	result->x = a->y * b->z - a->z * b->y;
	result->y = a->z * b->x - a->x * b->z;
	result->z = a->x * b->y - a->y * b->x;
}

// Initialize the model:
void
model_init (void)
{
	// Define our cube:
	// yash changes
	// struct cube cube =
	// { .face[0].corner =
	//   { { 0, 1, 0 }
	//   , { 1, 0, 0 }
	//   , { 0, 0, 0 }
	//   , { 1, 1, 0 }
	//   }
	// , .face[1].corner =
	//   { { 0, 0, 0 }
	//   , { 1, 0, 1 }
	//   , { 0, 0, 1 }
	//   , { 1, 0, 0 }
	//   }
	// , .face[2].corner =
	//   { { 1, 0, 0 }
	//   , { 1, 1, 1 }
	//   , { 1, 0, 1 }
	//   , { 1, 1, 0 }
	//   }
	// , .face[3].corner =
	//   { { 1, 1, 0 }
	//   , { 0, 1, 1 }
	//   , { 1, 1, 1 }
	//   , { 0, 1, 0 }
	//   }
	// , .face[4].corner =
	//   { { 0, 1, 0 }
	//   , { 0, 0, 1 }
	//   , { 0, 1, 1 }
	//   , { 0, 0, 0 }
	//   }
	// , .face[5].corner =
	//   { { 0, 1, 1 }
	//   , { 1, 0, 1 }
	//   , { 1, 1, 1 }
	//   , { 0, 0, 1 }
	//   }
	// } ;
	struct cube cube;
	// cube.face[0].corner[0]={{ 0, 1, 0 },{0,0,0}};
	// cube.face[0].corner[1]={{ 1, 0, 0 },{0,0,0}};
	// cube.face[0].corner[2]={{ 0, 0, 0 },{0,0,0}};
	// cube.face[0].corner[3]={{ 1, 1, 0 },{0,0,0}};

	// cube.face[1].corner[0]={{ 0, 0, 0 },{0,0,0}};
	// cube.face[1].corner[1]={{ 1, 0, 1 },{0,0,0}};
	// cube.face[1].corner[2]={{ 0, 0, 1 },{0,0,0}};
	// cube.face[1].corner[3]={{ 1, 0, 0 },{0,0,0}};

	// cube.face[2].corner[0]={{ 1, 0, 0 },{0,0,0}};
	// cube.face[2].corner[1]={{ 1, 1, 1 },{0,0,0}};
	// cube.face[2].corner[2]={{ 1, 0, 1 },{0,0,0}};
	// cube.face[2].corner[3]={{ 1, 1, 0 },{0,0,0}};

	// cube.face[3].corner[0]={{ 1, 1, 0 },{0,0,0}};
	// cube.face[3].corner[1]={{ 0, 1, 1 },{0,0,0}};
	// cube.face[3].corner[2]={{ 1, 1, 1 },{0,0,0}};
	// cube.face[3].corner[3]={{ 0, 1, 0 },{0,0,0}};

	// cube.face[4].corner[0]={{ 0, 1, 0 },{0,0,0}};
	// cube.face[4].corner[1]={{ 0, 0, 1 },{0,0,0}};
	// cube.face[4].corner[2]={{ 0, 1, 1 },{0,0,0}};
	// cube.face[4].corner[3]={{ 0, 0, 0 },{0,0,0}};

	// cube.face[5].corner[0]={{ 0, 1, 1 },{0,0,0}};
	// cube.face[5].corner[1]={{ 1, 0, 1 },{0,0,0}};
	// cube.face[5].corner[2]={{ 1, 1, 1 },{0,0,0}};
	// cube.face[5].corner[3]={{ 0, 0, 1 },{0,0,0}};

	cube.face[0].corner[0]={{ -1, 1, -1 },{0,0,0}};
	cube.face[0].corner[1]={{ 1, -1, -1 },{0,0,0}};
	cube.face[0].corner[2]={{ -1, -1, -1 },{0,0,0}};
	cube.face[0].corner[3]={{ 1, 1, 0 },{0,0,0}};

	cube.face[1].corner[0]={{ -1, -1, -1 },{0,0,0}};
	cube.face[1].corner[1]={{ 1, -1, 1 },{0,0,0}};
	cube.face[1].corner[2]={{ -1, -1, 1 },{0,0,0}};
	cube.face[1].corner[3]={{ 1, -1, -1 },{0,0,0}};

	cube.face[2].corner[0]={{ 1, -1, -1 },{0,0,0}};
	cube.face[2].corner[1]={{ 1, 1, 1 },{0,0,0}};
	cube.face[2].corner[2]={{ 1, -1, 1 },{0,0,0}};
	cube.face[2].corner[3]={{ 1, 1, 0 },{0,0,0}};

	cube.face[3].corner[0]={{ 1, 1, -1 },{0,0,0}};
	cube.face[3].corner[1]={{ -1, 1, 1 },{0,0,0}};
	cube.face[3].corner[2]={{ 1, 1, 1 },{0,0,0}};
	cube.face[3].corner[3]={{ -1, 1, -1 },{0,0,0}};

	cube.face[4].corner[0]={{ -1, 1, -1 },{0,0,0}};
	cube.face[4].corner[1]={{ -1, -1, 1 },{0,0,0}};
	cube.face[4].corner[2]={{ -1, 1, 1 },{0,0,0}};
	cube.face[4].corner[3]={{ -1, -1, -1 },{0,0,0}};

	cube.face[5].corner[0]={{ -1, 1, 1 },{0,0,0}};
	cube.face[5].corner[1]={{ 1, -1, 1 },{0,0,0}};
	cube.face[5].corner[2]={{ 1, 1, 1 },{0,0,0}};
	cube.face[5].corner[3]={{ -1, -1, 1 },{0,0,0}};

	// Generate colors for each corner based on its position:
	FOREACH (cube.face, face) {
		FOREACH (face->corner, corner) {
			corner->color.r = corner->pos.x * 0.8f + 0.1f;
			corner->color.g = corner->pos.y * 0.8f + 0.1f;
			corner->color.b = corner->pos.z * 0.8f + 0.1f;
		}
	}

	// Center cube on the origin by translating corner points:
	// FOREACH (cube.face, face) {
	// 	FOREACH (face->corner, corner) {
	// 		corner->pos.x -= 0.0f;
	// 		corner->pos.y -= 0.0f;
	// 		corner->pos.z -= 0.0f;
	// 	}
	// }
	// Yash Change
	FOREACH (cube.face, face) {
		FOREACH (face->corner, corner) {
			corner->pos.x -= 0.5f;
			corner->pos.y -= 0.5f;
			corner->pos.z -= 0.5f;
		}
	}

	// Face normals are cross product of two ribs:
	FOREACH (cube.face, face) {

		// First rib is (corner 3 - corner 0):
		struct point a = {
			.x = face->corner[3].pos.x - face->corner[0].pos.x,
			.y = face->corner[3].pos.y - face->corner[0].pos.y,
			.z = face->corner[3].pos.z - face->corner[0].pos.z,
		};

		// Second rib is (corner 2 - corner 0):
		struct point b = {
			.x = face->corner[2].pos.x - face->corner[0].pos.x,
			.y = face->corner[2].pos.y - face->corner[0].pos.y,
			.z = face->corner[2].pos.z - face->corner[0].pos.z,
		};

		// Face normal is cross product of these two ribs:
		cross(&face->normal, &a, &b);
	}

	// Create two triangles for each face:
	FOREACH (cube.face, face) {

		// Corners to compose triangles of, chosen in
		// such a way that both triangles rotate CCW:
		int index[2][3] = { { 0, 2, 1 }, { 1, 3, 0 } };

		for (int t = 0; t < 2; t++) {
			for (int v = 0; v < 3; v++) {
				int c = index[t][v];
				struct corner *corner = &face->corner[c];
				struct vertex *vertex = &face->tri[t].vert[v];

				vertex->pos = corner->pos;
				vertex->normal = face->normal;
				vertex->color = corner->color;
			}
		}
	}

	// Copy vertices into separate array for drawing:
	struct vertex vertex[6 * 2 * 3];
	struct vertex *cur = vertex;

	FOREACH (cube.face, face) {
		FOREACH (face->tri, tri) {
			for (int v = 0; v < 3; v++) {
				*cur++ = tri->vert[v];
			}
		}
	}

	// Generate empty buffer:
	glGenBuffers(1, &vbo);

	// Generate empty vertex array object:
	glGenVertexArrays(1, &vao);

	// Set as current vertex array:
	glBindVertexArray(vao);
	glBindBuffer(GL_ARRAY_BUFFER, vbo);

	// Add vertex, color and normal data to buffers:
	struct {
		enum LocCube	 loc;
		const void	*ptr;
	}
	map[] = {
		{ .loc = LOC_CUBE_VERTEX
		, .ptr = (void *) offsetof(struct vertex, pos)
		} ,
		{ .loc = LOC_CUBE_VCOLOR
		, .ptr = (void *) offsetof(struct vertex, color)
		} ,
		{ .loc = LOC_CUBE_NORMAL
		, .ptr = (void *) offsetof(struct vertex, normal)
		} ,
	};

	FOREACH (map, m) {
		GLint loc = program_cube_loc(m->loc);
		glEnableVertexAttribArray(loc);
		glVertexAttribPointer(loc, 3, GL_FLOAT, GL_FALSE, sizeof(struct vertex), m->ptr);
	}

	// Upload vertex data:
	glBufferData(GL_ARRAY_BUFFER, sizeof(vertex), vertex, GL_STATIC_DRAW);
}

static GLuint matToTexture(unsigned char* buffer , GLenum minFilter, GLenum magFilter, GLenum wrapFilter, int width, int height) {
    // Generate a number for our textureID's unique handle
    GLuint textureID;
    glGenTextures(1, &textureID);

    // Bind to our texture handle
    glBindTexture(GL_TEXTURE_2D, textureID);

    // Catch silly-mistake texture interpolation method for magnification
    if (magFilter == GL_LINEAR_MIPMAP_LINEAR  ||
            magFilter == GL_LINEAR_MIPMAP_NEAREST ||
            magFilter == GL_NEAREST_MIPMAP_LINEAR ||
            magFilter == GL_NEAREST_MIPMAP_NEAREST)
    {
		// printf("You can't use MIPMAPs for magnification - setting filter to GL_LINEAR\n");
        std::cout << "You can't use MIPMAPs for magnification - setting filter to GL_LINEAR" << std::endl;
        magFilter = GL_LINEAR;
    }

    // Set texture interpolation methods for minification and magnification
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, minFilter);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, magFilter);

    // Set texture clamping method
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, wrapFilter);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, wrapFilter);

    // Set incoming texture format to:
    // GL_BGR       for CV_CAP_OPENNI_BGR_IMAGE,
    // GL_LUMINANCE for CV_CAP_OPENNI_DISPARITY_MAP,
    // Work out other mappings as required ( there's a list in comments in main() )
    GLenum inputColourFormat = GL_RGBA;// GL_BGR
    if (3 == 1)
    {
        inputColourFormat = GL_LUMINANCE;
    }
	GLint max_texture_size;
	glGetIntegerv(GL_MAX_TEXTURE_SIZE, &max_texture_size);
	// std::cout << "Maximum texture size: " << max_texture_size << std::endl;
    // Create the texture
	glTexImage2D(GL_TEXTURE_2D,		// Type of texture
				 0,					// Pyramid level (for mip-mapping) - 0 is the top level
				 GL_RGBA,			// CHanged from rgb to rgba Internal colour format to convert to
				 width,				// Image width  i.e. 640 for Kinect in standard mode
				 height,			// Image height i.e. 480 for Kinect in standard mode
				 0,					// Border width in pixels (can either be 1 or 0)
				 inputColourFormat, // Input image format (i.e. GL_RGB, GL_RGBA, GL_BGR etc.)
				 GL_UNSIGNED_BYTE,	// Image data type
				 buffer);			// The actual image data itself

	// If we're using mipmaps then generate them. Note: This requires OpenGL 3.0 or higher
    if (minFilter == GL_LINEAR_MIPMAP_LINEAR  ||
            minFilter == GL_LINEAR_MIPMAP_NEAREST ||
            minFilter == GL_NEAREST_MIPMAP_LINEAR ||
            minFilter == GL_NEAREST_MIPMAP_NEAREST)
    {
		// printf("Will Generate MinMap \n");
        // std::cout << "Will Generate minmap" << std::endl;
        glGenerateMipmap(GL_TEXTURE_2D);
    }

    return textureID;
}


// void 
// draw_frames(void)
// {
// 	unsigned char* buffer = (unsigned char*) malloc(640 * 480 * 3);
// 	    // Fill the buffer with green color
//     for (int i = 0; i < 640 * 480 * 3; i += 3) {
//         buffer[i] = 0;   // Blue channel
//         buffer[i + 1] = 255; // Green channel
//         buffer[i + 2] = 0;  // Red channel
//     }
// 	int window_height = 480;
// 	int window_width = 640;

// 	static float angle = 0.0f;

// 	// Rotate slightly:
// 	angle += 0.00f;

// 	// Setup rotation matrix:
// 	mat_rotate(matrix, rot.x, rot.y, rot.z, angle);

// 	// cv::Mat frame(480, 640, CV_8UC3, cv::Scalar(0, 255, 0));
// 	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
//     glMatrixMode(GL_MODELVIEW);     // Operate on model-view matrix

//     glEnable(GL_TEXTURE_2D);
// 	GLuint image_tex = matToTexture(buffer, GL_LINEAR_MIPMAP_LINEAR, GL_LINEAR, GL_CLAMP);
// 	glBegin(GL_QUADS);
//     glTexCoord2i(0, 0); glVertex2i(0,   0);
//     glTexCoord2i(0, 1); glVertex2i(0,   window_height);
//     glTexCoord2i(1, 1); glVertex2i(window_width, window_height);
//     glTexCoord2i(1, 0); glVertex2i(window_width, 0);
//     glEnd();

//     glDeleteTextures(1, &image_tex);
//     glDisable(GL_TEXTURE_2D);
// }
int global_count = 0;

void 
draw_frames(void)
{
	// unsigned char* buffer = (unsigned char*) malloc(640 * 480 * 3);
	//     // Fill the buffer with green color

	// if(global_count % 2 == 0)
	// {
	// 	for (int i = 0; i < 640 * 480 * 3; i += 3) {
    // 	    buffer[i] = 0;   // Blue channel
    //     	buffer[i + 1] = 255; // Green channel
    //     	buffer[i + 2] = 0;  // Red channel
    // 	}
	// }
	// else
	// {
	// 	for (int i = 0; i < 640 * 480 * 3; i += 3) {
    // 	    buffer[i] = 255;   // Blue channel
    //     	buffer[i + 1] = 0; // Green channel
    //     	buffer[i + 2] = 0;  // Red channel
    // 	}

	// }
	unsigned char *buffer = (unsigned char *)malloc(640 * 480 * 3);
	// Fill the buffer with green color

	if (global_count % 2 == 0)
	{
		for (int i = 0; i < 640 * 480 * 3; i += 3)
		{
			buffer[i] = 0;		 // Blue channel
			buffer[i + 1] = 255; // Green channel
			buffer[i + 2] = 0;	 // Red channel
		}
	}
	else
	{
		for (int i = 0; i < 640 * 480 * 3; i += 3)
		{
			buffer[i] = 255;   // Blue channel
			buffer[i + 1] = 0; // Green channel
			buffer[i + 2] = 0; // Red channel
		}
	}


	int window_height = 480;
	int window_width = 640;

	static float angle = 0.0f;

	// Rotate slightly:
	angle += 0.00f;

	// Setup rotation matrix:
	mat_rotate(matrix, rot.x, rot.y, rot.z, angle);
	
	// GLuint image_tex = matToTexture(buffer, GL_LINEAR_MIPMAP_LINEAR, GL_LINEAR, GL_CLAMP);
	
	// Don't clip against background:
	glClear(GL_DEPTH_BUFFER_BIT);

	// Draw all the triangles in the buffer:
	glBindVertexArray(vao);
	glDrawArrays(GL_TRIANGLES, 0, 12 * 3);
	global_count++;
}

void drawCameraFrame(void* frameData, int width, int height){

	static float angle = 0.0f;

	mat_rotate(matrix, rot.x, rot.y, rot.z, angle);
	
	// GLuint image_tex = matToTexture((unsigned char*)frameData, GL_LINEAR_MIPMAP_LINEAR, GL_LINEAR, GL_CLAMP, width, height);
	GLuint image_tex = matToTexture((unsigned char*)frameData, GL_LINEAR_MIPMAP_LINEAR, GL_LINEAR, GL_CLAMP_TO_BORDER, width, height);
	
	// Don't clip against background:
	glClear(GL_DEPTH_BUFFER_BIT);

	// Draw all the triangles in the buffer:
	glBindVertexArray(vao);
	glDrawArrays(GL_TRIANGLES, 0, 12 * 3);

	glDeleteTextures(1, &image_tex);
    glDisable(GL_TEXTURE_2D);

}

void
model_draw (void)
{
	static float angle = 0.0f;

	// Rotate slightly:
	angle += 0.00f;

	// Setup rotation matrix:
	mat_rotate(matrix, rot.x, rot.y, rot.z, angle);

	// Use our own shaders:
	// Main program for rendering 
	program_cube_use();

	// Don't clip against background:
	glClear(GL_DEPTH_BUFFER_BIT);

	// Draw all the triangles in the buffer:
	glBindVertexArray(vao);
	glDrawArrays(GL_TRIANGLES, 0, 12 * 3);
}

const float *
model_matrix (void)
{
	return matrix;
}

void
model_pan_start (int x, int y)
{
	pan.x = x;
	pan.y = y;
}

void
model_pan_move (int x, int y)
{
	int dx = pan.x - x;
	int dy = pan.y - y;

	// Rotation vector is perpendicular to (dx, dy):
	rot.x = dy;
	rot.y = -dx;
}
