#include <gdk/gdk.h>
#ifndef GL_H
#define GL_H
#include <GL/glew.h>
#include <GL/glut.h>
#endif
#include "GTKSetup.h"

static GLuint texture;
static GLuint vao, vbo;

// Each vertex has space and texture coordinates:
struct vertex {
	float x;
	float y;
	float u;
	float v;
} __attribute__((packed));

void
background_set_window (int width, int height)
{
	// The background quad is made of four vertices:
	//
	//   3--2
	//   |  |
	//   0--1
	//
	struct vertex vertex[4] = {
		{ -1, -1,  0,  0 },	// Bottom left
		{  1, -1, 1,  0 },	// Bottom right
		{  1,  1, 1, 1 },	// Top right
		{ -1,  1,  0, 1 },	// Top left
	};

	GLint loc_vertex  = program_bkgd_loc(LOC_BKGD_VERTEX);
	GLint loc_texture = program_bkgd_loc(LOC_BKGD_TEXTURE);

	glBindVertexArray(vao);

	glEnableVertexAttribArray(loc_vertex);
	glEnableVertexAttribArray(loc_texture);

	glBindBuffer(GL_ARRAY_BUFFER, vbo);
	glBufferData(GL_ARRAY_BUFFER, sizeof(vertex), vertex, GL_STATIC_DRAW);

	glVertexAttribPointer(loc_vertex, 2, GL_FLOAT, GL_FALSE,
		sizeof(struct vertex),
		(void *) offsetof(struct vertex, x));

	glVertexAttribPointer(loc_texture, 2, GL_FLOAT, GL_FALSE,
		sizeof(struct vertex),
		(void *) offsetof(struct vertex, u));

	glBindVertexArray(0);
}

void
background_draw (void)
{
	// Array of indices. We define two counterclockwise triangles:
	// 0-2-3 and 2-0-1
	//yash change
	// static GLubyte index[6] = {
	// 	0, 1, 1,
	// 	2, 0, 1,
	// 	1, 3, 0
	// };
	static GLubyte triangle1[] = {
    0, 1, 2,
    0, 2, 3
	};


	program_bkgd_use();
	// glActiveTexture(GL_TEXTURE0);
	// glBindTexture(GL_TEXTURE_2D, texture);
	glBindVertexArray(vao);
	// //yash change
	// // glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_BYTE, index);
	glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_BYTE, triangle1);
}

void
background_init (void)
{
	// Inline data declaration:
	// extern char _binary_textures_background_png_start[];
	// extern char _binary_textures_background_png_end[];

	// char *start = _binary_textures_background_png_start;
	// size_t len = _binary_textures_background_png_end
	// 	   - _binary_textures_background_png_start;

	char *start ="start";
	size_t len = strlen(start);

	GInputStream *stream;
	GdkPixbuf *pixbuf;

	// Create an input stream from inline data:
	stream = g_memory_input_stream_new_from_data(start, len, NULL);

	// Generate a pixbuf from the input stream:
	pixbuf = gdk_pixbuf_new_from_stream(stream, NULL, NULL);

	// Destroy the stream:
	g_object_unref(stream);

	// Generate an OpenGL texture from pixbuf;
	// hack a bit by not accounting for pixbuf rowstride:
	glGenTextures(1, &texture);
	glBindTexture(GL_TEXTURE_2D, texture);

	// glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA,
	// 	gdk_pixbuf_get_width(pixbuf),
	// 	gdk_pixbuf_get_height(pixbuf), 0, GL_RGBA, GL_UNSIGNED_BYTE,
	// 	gdk_pixbuf_get_pixels(pixbuf));

	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);

	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);

	// Generate empty buffer:
	glGenBuffers(1, &vbo);

	// Generate empty vertex array object:
	glGenVertexArrays(1, &vao);
}
