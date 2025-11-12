/*
 * Copyright (c) 2016, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include "ApraNvEglRenderer.h"
#include "NvLogging.h"
//#include "nvbuf_utils.h"
#include "NvElement.h"
#include <cstring>
#include <sys/time.h>
#include "Logger.h"
#include <X11/Xatom.h>
#include <ft2build.h>
#include FT_FREETYPE_H
#include <map>
#include <string>
#include <algorithm>
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>


struct Character {
    GLuint TextureID;   // Glyph texture
    int SizeX;          // Width
    int SizeY;          // Height
    int BearingX;       // Offset from baseline to left/top
    int BearingY;
    GLuint Advance;     // Offset to advance to next glyph
};

static std::map<char, Character> Characters;
static GLuint textVAO = 0, textVBO = 0;
static GLuint textShader = 0;
GLuint imageShader;
static int g_fontAscent = 0;
static int g_fontDescent = 0;
static int g_fontLineAdvance = 0;
static bool g_fontInitialized = false;
static std::string g_loadedFontPath;
static int g_loadedFontSize = 0;

#define CAT_NAME "EglRenderer"

#define ERROR_GOTO_FAIL(val, string) \
    do { \
        if (val) {\
            goto fail; \
        } \
    } while (0)

PFNEGLCREATEIMAGEKHRPROC                NvEglRenderer::eglCreateImageKHR;
PFNEGLDESTROYIMAGEKHRPROC               NvEglRenderer::eglDestroyImageKHR;
PFNEGLCREATESYNCKHRPROC                 NvEglRenderer::eglCreateSyncKHR;
PFNEGLDESTROYSYNCKHRPROC                NvEglRenderer::eglDestroySyncKHR;
PFNEGLCLIENTWAITSYNCKHRPROC             NvEglRenderer::eglClientWaitSyncKHR;
PFNEGLGETSYNCATTRIBKHRPROC              NvEglRenderer::eglGetSyncAttribKHR;
PFNGLEGLIMAGETARGETTEXTURE2DOESPROC     NvEglRenderer::glEGLImageTargetTexture2DOES;

PFNGLGENVERTEXARRAYSOESPROC             NvEglRenderer::glGenVertexArraysOES = nullptr;
PFNGLBINDVERTEXARRAYOESPROC             NvEglRenderer::glBindVertexArrayOES = nullptr;
PFNGLDELETEVERTEXARRAYSOESPROC          NvEglRenderer::glDeleteVertexArraysOES = nullptr;

using namespace std;

NvEglRenderer::NvEglRenderer(const char *name, uint32_t width, uint32_t height, uint32_t x_offset, uint32_t y_offset, const char *ttfFilePath,const char *message,float scale,float r,float g,float b,float fontsize,int textPosX, int textPosY,
                        string imagePath,int imagePosX,int imagePosY,uint32_t imageWidth,uint32_t imageHeight,float opacity,bool mask
                    ,float imageOpacity,float textOpacity) // alwaysOnTOp
{
    int depth;
    int screen_num;
    XSetWindowAttributes window_attributes;
    x_window = 0;
    x_display = NULL;

    texture_id = 0;
    cached_image_texture = 0;
    gc = NULL;
    fontinfo = NULL;

    egl_surface = EGL_NO_SURFACE;
    egl_context = EGL_NO_CONTEXT;
    egl_display = EGL_NO_DISPLAY;
    egl_config = NULL;

    memset(&last_render_time, 0, sizeof(last_render_time));
    stop_thread = false;
    render_thread = 0;
    render_fd = 0;

    memset(overlay_str, 0, sizeof(overlay_str));
    overlay_str_x_offset = 0;
    overlay_str_y_offset = 0;

    pthread_mutex_init(&render_lock, NULL);
    pthread_cond_init(&render_cond, NULL);

    setFPS(60);

    if (initEgl() < 0)
    {
        return;
    }

    x_display = XOpenDisplay(NULL);
    if (NULL == x_display)
    {
        return;
    }

    screen_num = DefaultScreen(x_display);
    if (!width || !height)
    {
        width = DisplayWidth(x_display, screen_num);
        height = DisplayHeight(x_display, screen_num);
        x_offset = 0;
        y_offset = 0;
    }


    depth = DefaultDepth(x_display, DefaultScreen(x_display));

    window_attributes.background_pixel =
        BlackPixel(x_display, DefaultScreen(x_display));

     window_attributes.override_redirect = 1;
    // window_attributes.override_redirect = 0; // this we added to make renderer controlled by window manager
    
    // Atom WM_HINTS; 
    // struct
    // {
    //     unsigned long flags;
    //     unsigned long functions;
    //     unsigned long decorations;
    //     long inputMode;
    //     unsigned long status;
    // } wmHints = { (1L << 1), 0, 0, 0, 0 }; // this we added to remove title bar

    // LOG_DEBUG << "X_OFFSET " << x_offset << "Y_OFFSET "<< y_offset << "========================>>>>";
    x_window = XCreateWindow(x_display,
                             DefaultRootWindow(x_display), x_offset,
                             y_offset, width, height,
                             0,
                             depth, CopyFromParent,
                             CopyFromParent,
                             (CWBackPixel | CWOverrideRedirect),
                             &window_attributes);
    
    // XStoreName(x_display, x_window, "ApraEglRenderer");
    // XFlush(x_display);
    
    // XSizeHints hints;
    // hints.x = x_offset;
    // hints.y = y_offset;
    // hints.width = width;
    // hints.height = height;
    // hints.flags = PPosition | PSize;
    // XSetWMNormalHints(x_display, x_window, &hints);

    // WM_HINTS = XInternAtom(x_display, "_MOTIF_WM_HINTS", True);
    // XChangeProperty(x_display, x_window, WM_HINTS, WM_HINTS, 32,
    //             PropModeReplace, (unsigned char *)&wmHints, 5);

    XSelectInput(x_display, (int32_t) x_window, ExposureMask);
    XMapWindow(x_display, (int32_t) x_window);
    gc = XCreateGC(x_display, x_window, 0, NULL);

    XSetForeground(x_display, gc,
                WhitePixel(x_display, DefaultScreen(x_display)) );
    fontinfo = XLoadQueryFont(x_display, "9x15bold");

    this->ttfFilePath = ttfFilePath ? std::string(ttfFilePath) : std::string();
    this->message = message ? std::string(message) : std::string();
    this->scale = scale;
    this->r = r;
    this->g = g;
    this->b = b;
    this->fontSize = fontsize;
    this->textPosX = textPosX;
    this->textPosY = textPosY; 
    this->opacity = opacity;
    this->imagePath = imagePath;
    this->imagePosX = imagePosX;
    this->imagePosY = imagePosY;
    this->imageWidth = imageWidth;
    this->imageHeight = imageHeight;
    this->mask = mask;
    this->imageOpacity = imageOpacity;
    this->textOpacity = textOpacity;

    if(opacity < 1.0f)
    {
        setWindowOpacity(opacity);
    }
    else
    {
    setWindowOpacity(0.99f);
    }
    pthread_mutex_lock(&render_lock);
    pthread_create(&render_thread, NULL, renderThread, this);
    pthread_setname_np(render_thread, "EglRenderer");
    pthread_cond_wait(&render_cond, &render_lock);
    pthread_mutex_unlock(&render_lock);
    return;
}

int
NvEglRenderer::getDisplayResolution(uint32_t &width, uint32_t &height)
{
    int screen_num;
    Display * x_display = XOpenDisplay(NULL);
    if (NULL == x_display)
    {
        return  -1;
    }

    screen_num = DefaultScreen(x_display);
    width = DisplayWidth(x_display, screen_num);
    height = DisplayHeight(x_display, screen_num);

    XCloseDisplay(x_display);
    x_display = NULL;

    return 0;
}

void *
NvEglRenderer::renderThread(void *arg)
{
    EGLBoolean egl_status;
    NvEglRenderer *renderer = (NvEglRenderer *) arg;

    static EGLint rgba8888[] = {
        EGL_RED_SIZE, 8,
        EGL_GREEN_SIZE, 8,
        EGL_BLUE_SIZE, 8,
        EGL_ALPHA_SIZE, 8,
        EGL_STENCIL_SIZE, 8,
        EGL_SURFACE_TYPE, EGL_WINDOW_BIT,
        EGL_NONE,
    };
    int num_configs = 0;
    EGLint context_attribs[] = { EGL_CONTEXT_CLIENT_VERSION, 2, EGL_NONE };
    renderer->egl_display = eglGetDisplay(renderer->x_display);
    if (EGL_NO_DISPLAY == renderer->egl_display)
    {
        goto error;
    }

    egl_status = eglInitialize(renderer->egl_display, 0, 0);
    if (!egl_status)
    {
        goto error;
    }

    egl_status =
        eglChooseConfig(renderer->egl_display, rgba8888,
                            &renderer->egl_config, 1, &num_configs);
    if (!egl_status)
    {
        goto error;
    }

    renderer->egl_context =
        eglCreateContext(renderer->egl_display, renderer->egl_config,
                            EGL_NO_CONTEXT, context_attribs);
    if (eglGetError() != EGL_SUCCESS)
    {
        goto error;
    }
    renderer->egl_surface =
        eglCreateWindowSurface(renderer->egl_display, renderer->egl_config,
                (EGLNativeWindowType) renderer->x_window, NULL);
    if (renderer->egl_surface == EGL_NO_SURFACE)
    {
        goto error;
    }

    eglMakeCurrent(renderer->egl_display, renderer->egl_surface,
                    renderer->egl_surface, renderer->egl_context);

    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    if (eglGetError() != EGL_SUCCESS)
    {
        goto error;
    }

    if (renderer->InitializeShaders() < 0)
    {
        goto error;
    }
    // Load OES VAO function pointers after making context current
    NvEglRenderer::glGenVertexArraysOES = (PFNGLGENVERTEXARRAYSOESPROC)eglGetProcAddress("glGenVertexArraysOES");
    NvEglRenderer::glBindVertexArrayOES = (PFNGLBINDVERTEXARRAYOESPROC)eglGetProcAddress("glBindVertexArrayOES");
    NvEglRenderer::glDeleteVertexArraysOES = (PFNGLDELETEVERTEXARRAYSOESPROC)eglGetProcAddress("glDeleteVertexArraysOES");
    
    textShader = renderer->initTextShader();
    // Initialize image shader used for 2D image rendering
    imageShader = renderer->initImageShader();
    // setup VAO/VBO for text quads
    if (glGenVertexArraysOES)
    {
        glGenVertexArraysOES(1, &textVAO);
    }
    glGenBuffers(1, &textVBO);
    if (glBindVertexArrayOES)
    {
        glBindVertexArrayOES(textVAO);
        glBindBuffer(GL_ARRAY_BUFFER, textVBO);
        glBufferData(GL_ARRAY_BUFFER, sizeof(float) * 6 * 4, NULL, GL_DYNAMIC_DRAW);
        glEnableVertexAttribArray(1);
        glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, 4 * sizeof(float), 0);
        glBindBuffer(GL_ARRAY_BUFFER, 0);
        glBindVertexArrayOES(0);
    }
    else
    {
        // No VAO support: just allocate buffer now; we'll set attrib pointer per draw
        glBindBuffer(GL_ARRAY_BUFFER, textVBO);
        glBufferData(GL_ARRAY_BUFFER, sizeof(float) * 6 * 4, NULL, GL_DYNAMIC_DRAW);
        glBindBuffer(GL_ARRAY_BUFFER, 0);
    }

    renderer->create_texture();

    // Load image texture once during initialization
    renderer->cached_image_texture = renderer->loadImageTexture(renderer->imagePath.c_str());
    if (renderer->cached_image_texture == 0)
    {
        fprintf(stderr, "Warning: Failed to load cached image texture\n");
    }

    // Initialize font atlas once if text is requested
    if (!renderer->message.empty() && !renderer->ttfFilePath.empty())
    {
        if (!g_fontInitialized || g_loadedFontPath != renderer->ttfFilePath || g_loadedFontSize != renderer->fontSize)
        {
            if (renderer->initFontAtlas(renderer->ttfFilePath.c_str(), renderer->fontSize) == 0)
            {
                g_fontInitialized = true;
                g_loadedFontPath = renderer->ttfFilePath;
                g_loadedFontSize = renderer->fontSize;
            }
            else
            {
                g_fontInitialized = false;
                fprintf(stderr, "ERROR: Font atlas initialization failed for '%s'\n", renderer->ttfFilePath.c_str());
            }
        }
    }

    pthread_mutex_lock(&renderer->render_lock);
    pthread_cond_broadcast(&renderer->render_cond);

    while (!renderer->stop_thread)
    {
        pthread_cond_wait(&renderer->render_cond, &renderer->render_lock);
        pthread_mutex_unlock(&renderer->render_lock);

        if(renderer->stop_thread)
        {
            pthread_mutex_lock(&renderer->render_lock);
            break;
        }

        renderer->renderInternal();

        pthread_mutex_lock(&renderer->render_lock);
        pthread_cond_broadcast(&renderer->render_cond);
    }
    pthread_mutex_unlock(&renderer->render_lock);

finish:
    if (renderer->texture_id)
    {
        glDeleteTextures(1, &renderer->texture_id);
    }
    if (renderer->cached_image_texture)
    {
        glDeleteTextures(1, &renderer->cached_image_texture);
    }

    if (renderer->egl_display != EGL_NO_DISPLAY)
    {
        eglMakeCurrent(renderer->egl_display, EGL_NO_SURFACE,
                EGL_NO_SURFACE, EGL_NO_CONTEXT);
    }

    if (renderer->egl_surface != EGL_NO_SURFACE)
    {
        egl_status = eglDestroySurface(renderer->egl_display,
                renderer->egl_surface);
        if (egl_status == EGL_FALSE)
        {
        }
    }

    if (renderer->egl_context != EGL_NO_CONTEXT)
    {
        egl_status = eglDestroyContext(renderer->egl_display,
                renderer->egl_context);
        if (egl_status == EGL_FALSE)
        {
            LOG_DEBUG << "GETTING EGL FALSE ";
        }
    }

    if (renderer->egl_display != EGL_NO_DISPLAY)
    {
        eglReleaseThread();
        eglTerminate(renderer->egl_display);
    }

    pthread_mutex_lock(&renderer->render_lock);
    pthread_cond_broadcast(&renderer->render_cond);
    pthread_mutex_unlock(&renderer->render_lock);
    return NULL;

error:
    goto finish;
}

NvEglRenderer::~NvEglRenderer()
{
    stop_thread = true;

    pthread_mutex_lock(&render_lock);
    pthread_cond_broadcast(&render_cond);
    pthread_mutex_unlock(&render_lock);

    pthread_join(render_thread, NULL);

    pthread_mutex_destroy(&render_lock);
    pthread_cond_destroy(&render_cond);

    if (fontinfo)
    {
        XFreeFont(x_display, fontinfo);
    }

    if (gc)
    {
        XFreeGC(x_display, gc);
    }

    if (x_window)
    {
        XUnmapWindow(x_display, (int32_t) x_window);
        XFlush(x_display);
        XDestroyWindow(x_display, (int32_t) x_window);
        x_window = 0;
    }
    if (x_display)
    {
        XCloseDisplay(x_display);
    }
}

int
NvEglRenderer::render(int fd)
{
    this->render_fd = fd;
    pthread_mutex_lock(&render_lock);
    pthread_cond_broadcast(&render_cond);
    pthread_cond_wait(&render_cond, &render_lock);
    pthread_mutex_unlock(&render_lock);
    return 0;
}

EGLImageKHR
NvEglRenderer::createEglImageFromDmaBuf()
{
    EGLAttrib attrs[32];
    int ai = 0;
    attrs[ai++] = EGL_WIDTH; attrs[ai++] = (EGLint)render_width;
    attrs[ai++] = EGL_HEIGHT; attrs[ai++] = (EGLint)render_height;
    attrs[ai++] = EGL_LINUX_DRM_FOURCC_EXT; attrs[ai++] = (EGLint)render_fourcc;
    // Plane 0
    attrs[ai++] = EGL_DMA_BUF_PLANE0_FD_EXT; attrs[ai++] = (EGLint)render_fd;
    attrs[ai++] = EGL_DMA_BUF_PLANE0_MODIFIER_LO_EXT; attrs[ai++] = 0;
    attrs[ai++] = EGL_DMA_BUF_PLANE0_MODIFIER_HI_EXT; attrs[ai++] = 0;
    attrs[ai++] = EGL_DMA_BUF_PLANE0_OFFSET_EXT; attrs[ai++] = (EGLint)render_offset;
    attrs[ai++] = EGL_DMA_BUF_PLANE0_PITCH_EXT; attrs[ai++] = (EGLint)render_pitch;
    if (render_num_planes >= 2)
    {
        attrs[ai++] = EGL_DMA_BUF_PLANE1_FD_EXT; attrs[ai++] = (EGLint)render_fd;
        attrs[ai++] = EGL_DMA_BUF_PLANE1_OFFSET_EXT; attrs[ai++] = (EGLint)render_offset1;
        attrs[ai++] = EGL_DMA_BUF_PLANE1_PITCH_EXT; attrs[ai++] = (EGLint)render_pitch1;
    }
    if (render_num_planes >= 3)
    {
        attrs[ai++] = EGL_DMA_BUF_PLANE2_FD_EXT; attrs[ai++] = (EGLint)render_fd;
        attrs[ai++] = EGL_DMA_BUF_PLANE2_OFFSET_EXT; attrs[ai++] = (EGLint)render_offset2;
        attrs[ai++] = EGL_DMA_BUF_PLANE2_PITCH_EXT; attrs[ai++] = (EGLint)render_pitch2;
    }
    attrs[ai++] = EGL_NONE;
    
    return eglCreateImage(
        egl_display,
        EGL_NO_CONTEXT,
        EGL_LINUX_DMA_BUF_EXT,
        (EGLClientBuffer)NULL,
        attrs
    );
}
int NvEglRenderer::renderVideoFrame(EGLImageKHR hEglImage)
{
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_EXTERNAL_OES, texture_id);
    glTexParameteri(GL_TEXTURE_EXTERNAL_OES, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_EXTERNAL_OES, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_EXTERNAL_OES, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_EXTERNAL_OES, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glEGLImageTargetTexture2DOES(GL_TEXTURE_EXTERNAL_OES, hEglImage);

    glUseProgram(gl_program);

    // --- Circle Mask Control ---
    GLint enableLoc = glGetUniformLocation(gl_program, "u_enableMask");
    GLint centerLoc = glGetUniformLocation(gl_program, "u_center");
    GLint radiusLoc = glGetUniformLocation(gl_program, "u_radius");

    if (enableLoc >= 0)  glUniform1i(enableLoc, mask ? 1 : 0);                 // enableMask is a bool member variable
    if (centerLoc >= 0)  glUniform2f(centerLoc, 0.5f, 0.5f);         // center at middle of frame
    if (radiusLoc >= 0)  glUniform1f(radiusLoc, 0.45f);              // circle radius (slightly less than half)

    glDrawArrays(GL_TRIANGLES, 0, 6);

    EGLBoolean egl_status = eglGetError();
    if (egl_status != EGL_SUCCESS)
        return -1;

    int iErr = glGetError();
    if (iErr != GL_NO_ERROR)
        return -1;

    return 0;
}


void
NvEglRenderer::saveGLState(GLint& prevProgram, GLint& prevVAO, GLint& prevTexExternal,
                           GLint& prevTex2D, GLint& prevArrayBuffer, GLint& prevActiveTexUnit,
                           GLboolean& wasBlendEnabled)
{
    wasBlendEnabled = glIsEnabled(GL_BLEND);
    glGetIntegerv(GL_CURRENT_PROGRAM, &prevProgram);
    if (glGetError() == GL_NO_ERROR && glBindVertexArrayOES)
    {
        glGetIntegerv(GL_VERTEX_ARRAY_BINDING_OES, &prevVAO);
    }
    glGetIntegerv(GL_ACTIVE_TEXTURE, &prevActiveTexUnit);
    glGetIntegerv(GL_TEXTURE_BINDING_EXTERNAL_OES, &prevTexExternal);
    glGetIntegerv(GL_TEXTURE_BINDING_2D, &prevTex2D);
    glGetIntegerv(GL_ARRAY_BUFFER_BINDING, &prevArrayBuffer);
}

void
NvEglRenderer::restoreGLState(GLint prevProgram, GLint prevVAO, GLint prevTexExternal,
                               GLint prevTex2D, GLint prevArrayBuffer, GLint prevActiveTexUnit,
                               GLboolean wasBlendEnabled)
{
    glUseProgram(prevProgram);
    if (glBindVertexArrayOES && prevVAO)
    {
        glBindVertexArrayOES(prevVAO);
    }
    glActiveTexture(prevActiveTexUnit);
    glBindTexture(GL_TEXTURE_2D, prevTex2D);
    glBindTexture(GL_TEXTURE_EXTERNAL_OES, prevTexExternal);
    glBindBuffer(GL_ARRAY_BUFFER, prevArrayBuffer);
    
    if (!wasBlendEnabled)
    {
        glDisable(GL_BLEND);
    }
    
    // If VAO is unavailable, restore video attribute pointer for 'in_pos'
    if (!glBindVertexArrayOES)
    {
        GLint inPosLoc = glGetAttribLocation(prevProgram, "in_pos");
        if (inPosLoc >= 0)
        {
            glVertexAttribPointer(inPosLoc, 4, GL_FLOAT, GL_FALSE, 0, (void*)0);
            glEnableVertexAttribArray(inPosLoc);
        }
    }
}
void
NvEglRenderer::renderOverlays()
{
    // Ensure depth/stencil won't block overlays and enable blending
    glDisable(GL_DEPTH_TEST);
    glDisable(GL_STENCIL_TEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    // Save GL state before rendering overlays
    GLint prevProgram;
    GLint prevVAO = 0;
    GLint prevTexExternal;
    GLint prevTex2D;
    GLint prevArrayBuffer;
    GLint prevActiveTexUnit;
    GLboolean wasBlendEnabled;
    saveGLState(prevProgram, prevVAO, prevTexExternal, prevTex2D, prevArrayBuffer,
                prevActiveTexUnit, wasBlendEnabled);

    GLint viewport[4] = {0, 0, 0, 0};
    glGetIntegerv(GL_VIEWPORT, viewport);
    int viewportWidth = viewport[2];
    int viewportHeight = viewport[3];
    if (viewportWidth <= 0 && render_width > 0)
    {
        viewportWidth = static_cast<int>(render_width);
    }
    if (viewportHeight <= 0 && render_height > 0)
    {
        viewportHeight = static_cast<int>(render_height);
    }
    float orthoWidth = viewportWidth > 0 ? static_cast<float>(viewportWidth)
                                         : (render_width > 0 ? render_width : 1.0f);
    float orthoHeight = viewportHeight > 0 ? static_cast<float>(viewportHeight)
                                           : (render_height > 0 ? render_height : 1.0f);

    // Render image overlay if available
    if (cached_image_texture != 0)
    {
        glUseProgram(imageShader);
        GLint projLoc = glGetUniformLocation(imageShader, "projection");
        if (projLoc != -1)
        {
            glm::mat4 projection = glm::ortho(0.0f, orthoWidth, orthoHeight, 0.0f);
            glUniformMatrix4fv(projLoc, 1, GL_FALSE, glm::value_ptr(projection));
        }
        RenderImage(cached_image_texture, imagePosX, imagePosY, imageWidth, imageHeight);
    }

    // Render text overlay if message is not empty
    if (!message.empty() && !ttfFilePath.empty())
    {
        if (!g_fontInitialized)
        {
            // Attempt lazy init if not already initialized
            if (initFontAtlas(ttfFilePath.c_str(), fontSize) == 0)
            {
                g_fontInitialized = true;
                g_loadedFontPath = ttfFilePath;
                g_loadedFontSize = fontSize;
            }
            else
            {
                fprintf(stderr, "ERROR: Skipping text render because font atlas is not initialized for '%s'\n", ttfFilePath.c_str());
                // Skip text rendering if font couldn't be initialized
                goto restore_state;
            }
        }
        // Text projection matrix (normalized device coordinates)
        GLfloat textProjection[16] = {
            2.0f / orthoWidth, 0.0f, 0.0f, 0.0f,
            0.0f, -2.0f / orthoHeight, 0.0f, 0.0f,
            0.0f, 0.0f, -1.0f, 0.0f,
            -1.0f, 1.0f, 0.0f, 1.0f
        };

        glUseProgram(textShader);
        GLint projLoc = glGetUniformLocation(textShader, "projection");
        if (projLoc != -1)
            glUniformMatrix4fv(projLoc, 1, GL_FALSE, textProjection);

        // Compute multiline-aware text dimensions
        int maxLineW = 0;
        int currentLineW = 0;
        int numLines = 1;
        for (auto it = message.begin(); it != message.end(); ++it)
        {
            if (*it == '\n')
            {
                if (currentLineW > maxLineW) maxLineW = currentLineW;
                currentLineW = 0;
                numLines++;
                continue;
            }
            auto cit = Characters.find(*it);
            if (cit == Characters.end())
            {
                // Approximate advance for missing glyphs (e.g., space)
                continue;
            }
            const Character& ch = cit->second;
            currentLineW += static_cast<int>(((int)ch.Advance >> 6) * scale);
        }
        if (currentLineW > maxLineW) maxLineW = currentLineW;
        int textW = maxLineW > 0 ? maxLineW : 1;
        int textH = (g_fontLineAdvance > 0 ? g_fontLineAdvance : 16) * numLines * scale;
        if (textH < 1) textH = 1;

        // Save previous scissor state
        GLboolean prevScissorEnabled = glIsEnabled(GL_SCISSOR_TEST);
        GLint prevScissorBox[4];
        glGetIntegerv(GL_SCISSOR_BOX, prevScissorBox);

        // Determine if stencil buffer is available before using it for clipping
        GLint stencilBits = 0;
        glGetIntegerv(GL_STENCIL_BITS, &stencilBits);
        const bool useStencil = stencilBits > 0;

        if (useStencil)
        {
            // Prepare stencil: 0 everywhere, 1 inside text rectangle
            glEnable(GL_STENCIL_TEST);
            glStencilMask(0xFF);
            glDisable(GL_SCISSOR_TEST);
            glClearStencil(0);
            glClear(GL_STENCIL_BUFFER_BIT);
        }

        // Compute scissor box for top-left coordinates to GL bottom-left origin
        int ascentPixels = g_fontAscent > 0 ? static_cast<int>(g_fontAscent * scale) : static_cast<int>(16 * scale);
        int adjustedtextPosY = static_cast<int>(textPosY) - ascentPixels;
        int adjustedTextH = textH + ascentPixels;
        if (adjustedtextPosY < 0)
        {
            adjustedTextH += adjustedtextPosY;
            adjustedtextPosY = 0;
        }

        GLint scX = static_cast<GLint>(textPosX);
        GLint scY = static_cast<GLint>(viewportHeight - adjustedtextPosY - adjustedTextH);
        if (scY < 0) scY = 0;
        if (scX < 0) scX = 0;
        GLsizei scW = static_cast<GLsizei>(textW);
        GLsizei scH = static_cast<GLsizei>(adjustedTextH);
        if (viewportWidth > 0)
        {
            if (scX + scW > viewportWidth)
            {
                int excess = (scX + scW) - viewportWidth;
                scW = static_cast<GLsizei>(std::max(1, static_cast<int>(scW) - excess));
            }
        }
        if (viewportHeight > 0)
        {
            if (scY + scH > viewportHeight)
            {
                int excess = (scY + scH) - viewportHeight;
                scH = static_cast<GLsizei>(std::max(1, static_cast<int>(scH) - excess));
            }
        }

        // Use scissor only to clip text rendering (no stencil)
        glEnable(GL_SCISSOR_TEST);
        glScissor(scX, scY, scW, scH);
        if (useStencil)
        {
            glClearStencil(1);
            glClear(GL_STENCIL_BUFFER_BIT);

            // Only pass where stencil == 1
            glStencilFunc(GL_EQUAL, 1, 0xFF);
            glStencilOp(GL_KEEP, GL_KEEP, GL_KEEP);
        }
        else
        {
            glDisable(GL_STENCIL_TEST);
        }

        // For non-VAO path, set attrib pointer for text
        if (!glBindVertexArrayOES)
        {
            glBindBuffer(GL_ARRAY_BUFFER, textVBO);
            glEnableVertexAttribArray(1);
            glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, 4 * sizeof(float), 0);
        }

        RenderText(this->message, this->textPosX, this->textPosY, this->scale, this->r, this->g, this->b);
        // Disable stencil and restore scissor state
        if (useStencil)
        {
            glDisable(GL_STENCIL_TEST);
        }
        if (prevScissorEnabled)
        {
            glEnable(GL_SCISSOR_TEST);
            glScissor(prevScissorBox[0], prevScissorBox[1], prevScissorBox[2], prevScissorBox[3]);
        }
        else
        {
            glDisable(GL_SCISSOR_TEST);
        }
    }

restore_state:
    // Restore GL state
    restoreGLState(prevProgram, prevVAO, prevTexExternal, prevTex2D, prevArrayBuffer,
                   prevActiveTexUnit, wasBlendEnabled);
}

int
NvEglRenderer::renderInternal()
{
    // Create EGL image from dmabuf
    EGLImageKHR hEglImage = createEglImageFromDmaBuf();
    if (!hEglImage)
    {
        return -1;
    }

    // Render video frame
    if (renderVideoFrame(hEglImage) < 0)
    {
        eglDestroyImageKHR(egl_display, hEglImage);
        return -1;
    }

    // Create sync object for frame timing
    EGLSyncKHR egl_sync = eglCreateSyncKHR(egl_display, EGL_SYNC_FENCE_KHR, NULL);
    if (egl_sync == EGL_NO_SYNC_KHR)
    {
        eglDestroyImageKHR(egl_display, hEglImage);
        return -1;
    }

    // Frame timing control
    if (last_render_time.tv_sec != 0)
    {
        pthread_mutex_lock(&render_lock);
        last_render_time.tv_sec += render_time_sec;
        last_render_time.tv_nsec += render_time_nsec;
        last_render_time.tv_sec += last_render_time.tv_nsec / 1000000000UL;
        last_render_time.tv_nsec %= 1000000000UL;
        pthread_cond_timedwait(&render_cond, &render_lock, &last_render_time);
        pthread_mutex_unlock(&render_lock);
    }
    else
    {
        struct timeval now;
        gettimeofday(&now, NULL);
        last_render_time.tv_sec = now.tv_sec;
        last_render_time.tv_nsec = now.tv_usec * 1000L;
    }

    // Render overlays (image and text)
    renderOverlays();

    // Swap buffers
    eglSwapBuffers(egl_display, egl_surface);
    if (eglGetError() != EGL_SUCCESS)
    {
        eglDestroySyncKHR(egl_display, egl_sync);
        eglDestroyImageKHR(egl_display, hEglImage);
        return -1;
    }

    // Wait for sync and cleanup
    eglClientWaitSyncKHR(egl_display, egl_sync, EGL_SYNC_FLUSH_COMMANDS_BIT_KHR, EGL_FOREVER_KHR);
    eglDestroySyncKHR(egl_display, egl_sync);
    eglDestroyImageKHR(egl_display, hEglImage);

    // Render X11 overlay text if needed
    if (strlen(overlay_str) != 0)
    {
        XSetForeground(x_display, gc, BlackPixel(x_display, DefaultScreen(x_display)));
        XSetFont(x_display, gc, fontinfo->fid);
        XDrawString(x_display, x_window, gc, overlay_str_x_offset,
                    overlay_str_y_offset, overlay_str, strlen(overlay_str));
    }

    return 0;
}

int
NvEglRenderer::setOverlayText(char *str, uint32_t x, uint32_t y)
{
    strncpy(overlay_str, str, sizeof(overlay_str));
    overlay_str[sizeof(overlay_str) - 1] = '\0';

    overlay_str_x_offset = x;
    overlay_str_y_offset = y;

    return 0;
}

int
NvEglRenderer::setFPS(float fps)
{
    LOG_DEBUG << "Setting FPS to =======>>>" << fps;
    uint64_t render_time_usec;

    if (fps == 0)
    {
        return -1;
    }
    pthread_mutex_lock(&render_lock);
    this->fps = fps;

    render_time_usec = 1000000L / fps;
    render_time_sec = render_time_usec / 1000000;
    render_time_nsec = (render_time_usec % 1000000) * 1000L;
    pthread_mutex_unlock(&render_lock);
    return 0;
}

NvEglRenderer *
NvEglRenderer::createEglRenderer(const char *name, uint32_t width,
                               uint32_t height, uint32_t x_offset,
                               uint32_t y_offset, const char *ttfFilePath,
                               const char *message, float scale, float r, float g, float b, float fontsize,int textPosX, int textPosY,
                            string imagePath,int imagePosX,int imagePosY,uint32_t imageWidth,uint32_t imageHeight,float opacity,bool mask
                        ,float imageOpacity,float textOpacity)
{
    NvEglRenderer* renderer = new NvEglRenderer(name, width, height,
                                    x_offset, y_offset, ttfFilePath, message, scale, r, g, b, fontsize, textPosX, textPosY, imagePath, imagePosX, imagePosY, imageWidth, imageHeight,opacity,mask,imageOpacity,textOpacity);
    return renderer;
}

int
NvEglRenderer::initEgl()
{
    eglCreateImageKHR =
        (PFNEGLCREATEIMAGEKHRPROC) eglGetProcAddress("eglCreateImageKHR");
    ERROR_GOTO_FAIL(!eglCreateImageKHR,
                    "ERROR getting proc addr of eglCreateImageKHR\n");

    eglDestroyImageKHR =
        (PFNEGLDESTROYIMAGEKHRPROC) eglGetProcAddress("eglDestroyImageKHR");
    ERROR_GOTO_FAIL(!eglDestroyImageKHR,
                    "ERROR getting proc addr of eglDestroyImageKHR\n");

    eglCreateSyncKHR =
        (PFNEGLCREATESYNCKHRPROC) eglGetProcAddress("eglCreateSyncKHR");
    ERROR_GOTO_FAIL(!eglCreateSyncKHR,
                    "ERROR getting proc addr of eglCreateSyncKHR\n");

    eglDestroySyncKHR =
        (PFNEGLDESTROYSYNCKHRPROC) eglGetProcAddress("eglDestroySyncKHR");
    ERROR_GOTO_FAIL(!eglDestroySyncKHR,
                    "ERROR getting proc addr of eglDestroySyncKHR\n");

    eglClientWaitSyncKHR =
        (PFNEGLCLIENTWAITSYNCKHRPROC) eglGetProcAddress("eglClientWaitSyncKHR");
    ERROR_GOTO_FAIL(!eglClientWaitSyncKHR,
                    "ERROR getting proc addr of eglClientWaitSyncKHR\n");

    eglGetSyncAttribKHR =
        (PFNEGLGETSYNCATTRIBKHRPROC) eglGetProcAddress("eglGetSyncAttribKHR");
    ERROR_GOTO_FAIL(!eglGetSyncAttribKHR,
                    "ERROR getting proc addr of eglGetSyncAttribKHR\n");

    glEGLImageTargetTexture2DOES =
        (PFNGLEGLIMAGETARGETTEXTURE2DOESPROC)
        eglGetProcAddress("glEGLImageTargetTexture2DOES");
    ERROR_GOTO_FAIL(!glEGLImageTargetTexture2DOES,
                    "ERROR getting proc addr of glEGLImageTargetTexture2DOES\n");

    return 0;

fail:
    return -1;
}

void
NvEglRenderer::CreateShader(GLuint program, GLenum type, const char *source,
        int size)
{

    char log[4096];
    int result = GL_FALSE;

    GLuint shader = glCreateShader(type);

    glShaderSource(shader, 1, &source, &size);
    glCompileShader(shader);
    glGetShaderiv(shader, GL_COMPILE_STATUS, &result);
    if (!result)
    {
        glGetShaderInfoLog(shader, sizeof(log), NULL, log);
    }
    glAttachShader(program, shader);

    if (glGetError() != GL_NO_ERROR)
    {
    }
}

int
NvEglRenderer::InitializeShaders(void)
{
    GLuint program;
    int result = GL_FALSE;
    char log[4096];
    uint32_t pos_location = 0;

    // pos_x, pos_y, uv_u, uv_v
    float vertexTexBuf[24] = {
        -1.0f, -1.0f, 0.0f, 1.0f,
        -1.0f,  1.0f, 0.0f, 0.0f,
         1.0f,  1.0f, 1.0f, 0.0f,
        -1.0f, -1.0f, 0.0f, 1.0f,
         1.0f,  1.0f, 1.0f, 0.0f,
         1.0f, -1.0f, 1.0f, 1.0f,
    };

    static const char kVertexShader[] = "varying vec2 interp_tc;\n"
        "attribute vec4 in_pos;\n"
        "void main() { \n"
        "interp_tc = in_pos.zw; \n" "gl_Position = vec4(in_pos.xy, 0, 1); \n" "}\n";

    static const char kFragmentShader[] =
        "#extension GL_OES_EGL_image_external : require\n"
        "precision mediump float;\n"
        "varying vec2 interp_tc;\n"
        "uniform samplerExternalOES tex;\n"
        "uniform int u_enableMask;\n"          // 0 = off, 1 = on
        "uniform vec2 u_center;\n"             // normalized [0,1] coordinates
        "uniform float u_radius;\n"            // normalized radius (0–1)\n"
        "void main() {\n"
        "    vec4 color = texture2D(tex, interp_tc);\n"
        "    if (u_enableMask == 1) {\n"
        "        float dist = distance(interp_tc, u_center);\n"
        "        if (dist > u_radius)\n"
        "            discard;\n"
        "    }\n"
        "    gl_FragColor = color;\n"
        "}\n";


    glEnable(GL_SCISSOR_TEST);
    program = glCreateProgram();

    CreateShader(program, GL_VERTEX_SHADER, kVertexShader,
                 sizeof(kVertexShader));
    CreateShader(program, GL_FRAGMENT_SHADER, kFragmentShader,
                 sizeof(kFragmentShader));

    glLinkProgram(program);
    if (glGetError() != GL_NO_ERROR)
    {
        return -1;
    }

    glGetProgramiv(program, GL_LINK_STATUS, &result);
    if (!result)
    {
        glGetShaderInfoLog(program, sizeof(log), NULL, log);
        return -1;
    }

    glUseProgram(program);
    if (glGetError() != GL_NO_ERROR)
    {
        return -1;
    }

    GLuint vbo; // Store vertex and tex coords
    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertexTexBuf), vertexTexBuf, GL_STATIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    pos_location = glGetAttribLocation(program, "in_pos");

    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glVertexAttribPointer(pos_location, 4, GL_FLOAT, GL_FALSE, 0, (void*)0);
    glEnableVertexAttribArray(pos_location);

    
    glActiveTexture(GL_TEXTURE0);
    // Bind the sampler used by the fragment shader to unit 0
    GLint texLoc = glGetUniformLocation(program, "tex");
    if (texLoc != -1)
    {
        glUniform1i(texLoc, 0);
    }

    this->alpha_location = 1;
    this->gl_program = program;

    if (glGetError() != GL_NO_ERROR)
    {
        return -1;
    }
    return 0;
}

int
NvEglRenderer::create_texture()
{
    int viewport[4];

    glGetIntegerv(GL_VIEWPORT, viewport);
    glViewport(viewport[0], viewport[1], viewport[2], viewport[3]);
    glScissor(viewport[0], viewport[1], viewport[2], viewport[3]);

    glGenTextures(1, &texture_id);

    glBindTexture(GL_TEXTURE_EXTERNAL_OES, texture_id);
    return 0;
}
void NvEglRenderer::setWindowOpacity(float opacity)
{
    if (opacity < 0.0f) opacity = 0.0f;
    if (opacity > 1.0f) opacity = 1.0f;

    unsigned long opacityValue = (unsigned long)(0xFFFFFFFFul * opacity);
    Atom opacityAtom = XInternAtom(x_display, "_NET_WM_WINDOW_OPACITY", False);
    XChangeProperty(x_display, x_window, opacityAtom, XA_CARDINAL, 32,
                    PropModeReplace, (unsigned char *)&opacityValue, 1);
    XFlush(x_display);
}

int NvEglRenderer::initFontAtlas(const char* fontPath, int fontSize)
{
    FT_Library ft;
    if (FT_Init_FreeType(&ft)) {
        fprintf(stderr, "ERROR: Could not init FreeType Library\n");
        return -1;
    }

    FT_Face face;
    if (FT_New_Face(ft, fontPath, 0, &face)) {
        fprintf(stderr, "ERROR: Failed to load font: %s\n", fontPath);
        return -1;
    }

    FT_Set_Pixel_Sizes(face, 0, fontSize);
    // Cache font metrics for baseline and line advance
    g_fontAscent = static_cast<int>(face->size->metrics.ascender >> 6);
    g_fontDescent = static_cast<int>(-(face->size->metrics.descender >> 6));
    g_fontLineAdvance = static_cast<int>(face->size->metrics.height >> 6);
    if (g_fontLineAdvance == 0)
    {
        g_fontLineAdvance = g_fontAscent + g_fontDescent;
    }
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1); // disable byte-alignment restriction

    for (unsigned char c = 0; c < 128; c++) {
        if (FT_Load_Char(face, c, FT_LOAD_RENDER)) {
            fprintf(stderr, "ERROR: Failed to load Glyph %c\n", c);
            continue;
        }

        GLuint tex;
        glGenTextures(1, &tex);
        glBindTexture(GL_TEXTURE_2D, tex);
        glTexImage2D(
            GL_TEXTURE_2D,
            0,
            GL_ALPHA,
            face->glyph->bitmap.width,
            face->glyph->bitmap.rows,
            0,
            GL_ALPHA,
            GL_UNSIGNED_BYTE,
            face->glyph->bitmap.buffer
        );

        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

        Character character = {
            tex,
            (int)face->glyph->bitmap.width,
            (int)face->glyph->bitmap.rows,
            (int)face->glyph->bitmap_left,
            (int)face->glyph->bitmap_top,
            (GLuint)face->glyph->advance.x
        };
        Characters.insert(std::pair<char, Character>(c, character));
    }

    FT_Done_Face(face);
    FT_Done_FreeType(ft);
    return 0;
}
GLuint NvEglRenderer::initTextShader()
{
    static const char* textVertexShaderSrc = R"(
    attribute vec4 vertex;
    varying vec2 TexCoords;
    uniform mat4 projection;
    void main() {
        gl_Position = projection * vec4(vertex.xy, 0.0, 1.0);
        TexCoords = vertex.zw;
    }
    )";

    static const char* textFragmentShaderSrc = R"(
    precision mediump float;
    varying vec2 TexCoords;
    uniform sampler2D text;
    uniform vec3 textColor;
    uniform float opacity;  
    void main() {
        float alpha = texture2D(text, TexCoords).a;
        vec4 color = vec4(textColor, alpha); 
        color.a *= opacity;                  
        gl_FragColor = color;               
    }
    )";

    GLuint vs = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vs, 1, &textVertexShaderSrc, NULL);
    glCompileShader(vs);

    GLuint fs = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fs, 1, &textFragmentShaderSrc, NULL);
    glCompileShader(fs);

    GLuint shader = glCreateProgram();
    glAttachShader(shader, vs);
    glAttachShader(shader, fs);
    glBindAttribLocation(shader, 1, "vertex");
    glLinkProgram(shader);

    // Bind uniforms
    glUseProgram(shader);
    GLint textSamplerLoc = glGetUniformLocation(shader, "text");
    GLint opacityLoc = glGetUniformLocation(shader, "opacity");
    if (textSamplerLoc != -1)
        glUniform1i(textSamplerLoc, 0); 
    if (opacityLoc != -1)
        glUniform1f(opacityLoc, textOpacity); 
    glUseProgram(0);

    glDeleteShader(vs);
    glDeleteShader(fs);
    return shader;
}

void NvEglRenderer::RenderText(std::string text, float x, float y, float scale, float r, float g, float b)
{
    // Assumes textShader is already bound and projection uniform set by caller
    GLint colorLoc = glGetUniformLocation(textShader, "textColor");
    glUniform3f(colorLoc, r, g, b);
    glActiveTexture(GL_TEXTURE0);
    if (glBindVertexArrayOES)
    {
        glBindVertexArrayOES(textVAO);
    }
    else
    {
        // No VAO: ensure attribute pointer is set to textVBO at attrib 1
        glBindBuffer(GL_ARRAY_BUFFER, textVBO);
        glEnableVertexAttribArray(1);
        glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, 4 * sizeof(float), 0);
    }

    const float originX = x;
    // In screen coordinates: y=0 is top, y increases downward
    // Baseline is at y position (y is top of text area)
    float baselineY = y;

    for (auto c = text.begin(); c != text.end(); c++) {
        if (*c == '\n')
        {
            // Move to next line (downward in screen space)
            x = originX;
            baselineY += g_fontLineAdvance * scale;
            continue;
        }
        auto it = Characters.find(*c);
        if (it == Characters.end())
        {
            // Skip unsupported glyphs
            continue;
        }
        const Character& ch = it->second;

        // Compute glyph position
        float xpos = x + ch.BearingX * scale;
        // ypos is the top of the glyph rectangle
        // BearingY is offset from baseline to top, so top = baseline - BearingY
        float ypos = baselineY - (ch.BearingY * scale);

        float w = ch.SizeX * scale;
        float h = ch.SizeY * scale;

        // Build vertex array (ypos = top, ypos + h = bottom)
       float vertices[6][4] = {
        {xpos,     ypos + h,   0.0f, 1.0f},  // flip V
        {xpos,     ypos,       0.0f, 0.0f},
        {xpos + w, ypos,       1.0f, 0.0f},
        {xpos,     ypos + h,   0.0f, 1.0f},
        {xpos + w, ypos,       1.0f, 0.0f},
        {xpos + w, ypos + h,   1.0f, 1.0f}
    };


        glBindTexture(GL_TEXTURE_2D, ch.TextureID);
        glBindBuffer(GL_ARRAY_BUFFER, textVBO);
        glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(vertices), vertices);
        glDrawArrays(GL_TRIANGLES, 0, 6);

        // Advance cursor
        x += (ch.Advance >> 6) * scale;
    }

    if (glBindVertexArrayOES)
    {
        glBindVertexArrayOES(0);
    }
    glBindTexture(GL_TEXTURE_2D, 0);
}

GLuint NvEglRenderer::loadImageTexture(const char* imagePath)
{
    int width, height, channels;
    unsigned char* data = stbi_load(imagePath, &width, &height, &channels, STBI_rgb_alpha);
    if (!data) {
        fprintf(stderr, "Failed to load image: %s\n", imagePath);
        return 0;
    }

    GLuint tex;
    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0,
                 GL_RGBA, GL_UNSIGNED_BYTE, data);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glBindTexture(GL_TEXTURE_2D, 0);

    stbi_image_free(data);
    return tex;
}

GLuint NvEglRenderer::initImageShader()
{
    static const char* vsSrc = R"(
    attribute vec4 vertex;
    varying vec2 TexCoords;
    uniform mat4 projection;
    void main() {
        gl_Position = projection * vec4(vertex.xy, 0.0, 1.0);
        TexCoords = vertex.zw;
    })";

    static const char* fsSrc = R"(
    precision mediump float;
    varying vec2 TexCoords;
    uniform sampler2D imageTex;
    uniform float opacity;
    void main() {
        vec4 color = texture2D(imageTex, TexCoords);
        color.a *= opacity;
        gl_FragColor = color;
    })";

    GLuint vs = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vs, 1, &vsSrc, NULL);
    glCompileShader(vs);

    GLuint fs = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fs, 1, &fsSrc, NULL);
    glCompileShader(fs);

    GLuint shader = glCreateProgram();
    glAttachShader(shader, vs);
    glAttachShader(shader, fs);
    glBindAttribLocation(shader, 1, "vertex");
    glLinkProgram(shader);

    glUseProgram(shader);
    GLint texLoc = glGetUniformLocation(shader, "imageTex");
    GLint opacityLoc = glGetUniformLocation(shader, "opacity");
    if (texLoc != -1) glUniform1i(texLoc, 0);
    if (opacityLoc != -1) glUniform1f(opacityLoc, imageOpacity);
    glUseProgram(0);

    glDeleteShader(vs);
    glDeleteShader(fs);
    return shader;
}

void NvEglRenderer::RenderImage(GLuint texture, float x, float y, float width, float height)
{
    glUseProgram(imageShader);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, texture);

    float vertices[6][4] = {
        {x,         y + height,  0.0f, 1.0f},
        {x,         y,           0.0f, 0.0f},
        {x + width, y,           1.0f, 0.0f},
        {x,         y + height,  0.0f, 1.0f},
        {x + width, y,           1.0f, 0.0f},
        {x + width, y + height,  1.0f, 1.0f}
    };

    glBindBuffer(GL_ARRAY_BUFFER, textVBO);
    glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(vertices), vertices);
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, 4 * sizeof(float), 0);
    glDrawArrays(GL_TRIANGLES, 0, 6);

    glBindTexture(GL_TEXTURE_2D, 0);
    glUseProgram(0);
}
