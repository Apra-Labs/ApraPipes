#include <iostream>
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
//yash change
// #include <GL/gl.h>
#ifndef GL_H
#define GL_H
#include <GL/glew.h>
#include <GL/glut.h>
// #include <GL/gl.h>
// #include <GL/glext.h>
#endif
#include <GL/gl.h>
#include "Model.h"
#include "View.h"
#include "Program.h"
#include "GLUtils.h"


const GLchar  *CUBE_VERTEX_SOURCE =
"#version 330\n"
"uniform mat4 view_matrix;\n"
"uniform mat4 model_matrix;\n"
"in vec3 vertex;\n"
"in vec3 vcolor;\n"
"in vec3 normal;\n"
"out vec3 fcolor;\n"
"out vec3 fpos;\n"
"out float fdot;\n"
"void main (void)\n"
"{\n"
"	vec4 modelspace = model_matrix * vec4(vertex, 1.0);\n"
"	gl_Position = view_matrix * modelspace;\n"
"	fcolor = vcolor;\n"
"	vec4 sight = vec4(0, 0, -1.0, 0.0);\n"
"	vec4 wnormal = model_matrix * vec4(normal, 0.0);\n"
"	fdot = dot(sight, wnormal);\n"
"	fpos = modelspace.xyz;\n"
"}\n";


const GLchar  *CUBE_FRAGMENT_SOURCE =
"#version 330\n"
"in vec3 fcolor;\n"
"in vec3 fpos;\n"
"in float fdot;\n"
"out vec4 fragcolor;\n"
"void main (void)\n"
"{\n"
"	if (!gl_FrontFacing)\n"
"		return;\n"
"	vec3 linear = pow(fcolor, vec3(1.0 / 2.2));\n"
"	float dst = distance(vec3(0, 0, 2), fpos) * 0.4;\n"
"	vec3 scaled = linear * vec3(fdot * dst);\n"
"	fragcolor = vec4(pow(scaled, vec3(2.2)), 0.0);\n"
"}\n";

const GLchar  *BKGD_VERTEX_SOURCE =
"#version 150\n"
"in vec2 vertex;\n"
"in vec2 texture;\n"
"out vec2 ftex;\n"
"void main (void)\n"
"{\n"
"	ftex = vec2(texture.x, 1.0 - texture.y);\n"
"	gl_Position = vec4(vertex, 0.5, 1.0);\n"
"}\n";
// "	ftex = texture;\n"
// const GLchar *BKGD_VERTEX_SOURCE =
//     "#version 150\n"
//     "in vec2 vertex;\n"
//     "in vec2 texture;\n"
//     "out vec2 ftex;\n"
//     "uniform mat4 view_matrix;\n"  // Include any necessary matrices
//     "uniform mat4 model_matrix;\n"
//     "void main (void)\n"
//     "{\n"
//     "    ftex = (model_matrix * vec4(texture, 0.0, 1.0)).xy;\n"
//     "    gl_Position = view_matrix * vec4(vertex, 0.5, 1.0);\n"
//     "}\n";


const GLchar  *BKGD_FRAGMENT_SOURCE =
"#version 150\n"
"uniform sampler2D tex;\n"
"in vec2 ftex;\n"
"out vec4 fragcolor;\n"
"void main (void)\n"
"{\n"
"	fragcolor = texture(tex, ftex);\n"
"};\n";



// Shader structure:
struct shader {
	const uint8_t	*buf;
	const uint8_t	*end;
	GLuint		 id;
};

// Location definitions:
enum loc_type {
	UNIFORM,
	ATTRIBUTE,
};

struct loc {
	const char	*name;
	enum loc_type	 type;
	GLint		 id;
};

static struct loc loc_bkgd[] = {
	[LOC_BKGD_VERTEX]  = { "vertex",	ATTRIBUTE },
	[LOC_BKGD_TEXTURE] = { "texture",	ATTRIBUTE },
};

static struct loc loc_cube[] = {
	[LOC_CUBE_VIEW]   = { "view_matrix",	UNIFORM   },
	[LOC_CUBE_MODEL]  = { "model_matrix",	UNIFORM   },
	[LOC_CUBE_VERTEX] = { "vertex",		ATTRIBUTE },
	[LOC_CUBE_VCOLOR] = { "vcolor",		ATTRIBUTE },
	[LOC_CUBE_NORMAL] = { "normal",		ATTRIBUTE },
};

// Programs:
enum {
	BKGD,
	CUBE,
};

struct program {
    struct {
        struct shader vert;
        struct shader frag;
    } shader;
    struct loc *loc;
    size_t nloc;
    GLuint id;
};

static program programs[2] = {
    {
        {
			{
				(const uint8_t *)BKGD_VERTEX_SOURCE,
				(const uint8_t *)BKGD_VERTEX_SOURCE + strlen(BKGD_VERTEX_SOURCE)
			},
			{
				(const uint8_t *)BKGD_FRAGMENT_SOURCE,
				(const uint8_t *)BKGD_FRAGMENT_SOURCE + strlen(BKGD_FRAGMENT_SOURCE)
			}
		},
        loc_bkgd,
        NELEM(loc_bkgd),
        0
    },
    {
		{
        	{
				(const uint8_t *)CUBE_VERTEX_SOURCE,
				(const uint8_t *)CUBE_VERTEX_SOURCE + strlen(CUBE_VERTEX_SOURCE)
			},
			{
				(const uint8_t *)CUBE_FRAGMENT_SOURCE,
				(const uint8_t *)CUBE_FRAGMENT_SOURCE + strlen(CUBE_FRAGMENT_SOURCE)
			}
		},
        loc_cube,
        NELEM(loc_cube),
        1
    }
};

static void
check_compile (GLuint shader)
{
	GLint length;

	glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &length);

	if (length <= 1)
		return;

	GLchar *log = (GLchar *)calloc(length, sizeof(GLchar));
	glGetShaderInfoLog(shader, length, NULL, log);
	fprintf(stderr, "glCompileShader failed:\n%s\n", log);
	free(log);
}

static void
check_link (GLuint program)
{
	GLint status, length;

	glGetProgramiv(program, GL_LINK_STATUS, &status);
	if (status != GL_FALSE)
		return;

	glGetProgramiv(program, GL_INFO_LOG_LENGTH, &length);
	GLchar *log = (GLchar *)calloc(length, sizeof(GLchar));
	glGetProgramInfoLog(program, length, NULL, log);
	fprintf(stderr, "glLinkProgram failed: %s\n", log);
	free(log);
}

static void
create_shader (struct shader *shader, GLenum type)
{
	auto x = glGetString(GL_SHADING_LANGUAGE_VERSION);
	GLenum err = glewInit();
	if (err != GLEW_OK)
	{
		std::cout << "GLEW IS NOT OK" << std::endl;
		exit(1); // or handle the error in a nicer way
	}
	if (!GLEW_VERSION_2_1)  // check that the machine supports the 2.1 API.
	{
		std::cout << "GLEW VERSION NOT SUPPORTED " << std::endl;
		exit(1); // or handle the error in a nicer way
		
	}
	const GLchar *buf = (const GLchar *) shader->buf;
	GLint len = shader->end - shader->buf;
	if (type == GL_FRAGMENT_SHADER){
		std::cout << "FRAGMENT _SHADERS " << std::endl;
		shader->id = glCreateShader(GL_FRAGMENT_SHADER);
	}
	else
	{
		std::cout << "VERTEX_SHADERS "<< GL_VERTEX_SHADER << std::endl;
		shader->id = glCreateShader(GL_VERTEX_SHADER);
	}
	
	glShaderSource(shader->id, 1, &buf, &len);
	glCompileShader(shader->id);

	check_compile(shader->id);
}

static void
program_init (struct program *p)
{
	struct shader *vert = &p->shader.vert;
	struct shader *frag = &p->shader.frag;

	create_shader(vert, GL_VERTEX_SHADER);
	create_shader(frag, GL_FRAGMENT_SHADER);

	p->id = glCreateProgram();

	glAttachShader(p->id, vert->id);
	glAttachShader(p->id, frag->id);

	glLinkProgram(p->id);
	check_link(p->id);

	glDetachShader(p->id, vert->id);
	glDetachShader(p->id, frag->id);

	glDeleteShader(vert->id);
	glDeleteShader(frag->id);

	FOREACH_NELEM (p->loc, p->nloc, l) {
		switch (l->type)
		{
		case UNIFORM:
			l->id = glGetUniformLocation(p->id, l->name);
			break;

		case ATTRIBUTE:
			l->id = glGetAttribLocation(p->id, l->name);
			break;
		}
	}
}

void
programs_init (void)
{
	FOREACH (programs, p)
		program_init(p);
}

void
program_cube_use (void)
{
	glUseProgram(programs[CUBE].id);

	glUniformMatrix4fv(loc_cube[LOC_CUBE_VIEW ].id, 1, GL_FALSE, view_matrix());
	glUniformMatrix4fv(loc_cube[LOC_CUBE_MODEL].id, 1, GL_FALSE, model_matrix());
}

void
program_bkgd_use (void)
{
	glUseProgram(programs[BKGD].id);

	glUniform1i(glGetUniformLocation(programs[BKGD].id, "tex"), 0);
}

GLint
program_bkgd_loc (const enum LocBkgd index)
{
	return loc_bkgd[index].id;
}

GLint
program_cube_loc (const enum LocCube index)
{
	return loc_cube[index].id;
}
