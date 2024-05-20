// source: https://github.com/aklomp/gtk3-opengl/blob/master/model.c
#include <stdlib.h>
#include <iostream>
#include <stddef.h>
#include <math.h>
//yash cahnge
// #include <GL/gl.h>
#ifndef GL_H
#define GL_H
#include "fstream"
#include <GL/glew.h>
#include <GL/glut.h>
#endif
// #include <opencv2/opencv.hpp>
#include "GTKMatrix.h"
#include "GTKSetup.h"
#include "GLUtils.h"

static GLuint vao, vbo;
static float matrix[16] = { 0 };


// Initialize the model:
void
model_init (void)
{
    glGenBuffers(1, &vbo);

    // Generate empty vertex array object:
    glGenVertexArrays(1, &vao);

    // Set as the current vertex array:
    glBindVertexArray(vao);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);

    // Vertices for a quad:
    GLfloat vertices[] = {
        -1.0f, -1.0f, 0.0f, 1.0f,
        1.0f, -1.0f, 0.0f, 1.0f,
        1.0f, 1.0f, 0.0f, 1.0f,
        -1.0f, 1.0f, 0.0f, 1.0f,
    };
	// Upload vertex data:
	// glBufferData(GL_ARRAY_BUFFER, sizeof(vertex), vertex, GL_STATIC_DRAW);
	glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

	    // Generate a number for our textureID's unique handle
    GLuint textureID;
    glGenTextures(1, &textureID);

    // Bind to our texture handle
    glBindTexture(GL_TEXTURE_2D, textureID);
}

void matToTexture(unsigned char* buffer , GLenum minFilter, GLenum magFilter, GLenum wrapFilter, int width, int height) {

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
    GLenum inputColourFormat = GL_RGB;// GL_BGR
    if (3 == 1)
    {
        inputColourFormat = GL_LUMINANCE;
    }

    // Create the texture
    glTexImage2D(GL_TEXTURE_2D,     // Type of texture
                 0,                 // Pyramid level (for mip-mapping) - 0 is the top level
                 GL_RGB,            // CHanged from rgb to rgba Internal colour format to convert to
                 width,          // Image width  i.e. 640 for Kinect in standard mode
                 height,          // Image height i.e. 480 for Kinect in standard mode
                 0,                 // Border width in pixels (can either be 1 or 0)
                 inputColourFormat, // Input image format (i.e. GL_RGB, GL_RGBA, GL_BGR etc.)
                 GL_UNSIGNED_BYTE,  // Image data type
                 buffer);        // The actual image data itself

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
}

void drawCameraFrame(void* frameData, int width, int height){

	static float angle = 0.0f;

	
	matToTexture((unsigned char*)frameData, GL_LINEAR_MIPMAP_LINEAR, GL_LINEAR, GL_CLAMP_TO_BORDER, width, height);
	
	// Don't clip against background:
	glClear(GL_DEPTH_BUFFER_BIT);

	// Draw all the triangles in the buffer:
	glBindVertexArray(vao);
    glDrawArrays(GL_QUADS, 0, 4);
}


const float *
model_matrix (void)
{
	return matrix;
}
