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
#include "Logger.h"
#include <cstring>
#include <sys/time.h>
#include "Logger.h"
#include <X11/Xatom.h>
#include <ft2build.h>
#include FT_FREETYPE_H
#include <map>
#include <string>

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
static int g_fontAscent = 0;
static int g_fontDescent = 0;
static int g_fontLineAdvance = 0;

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

NvEglRenderer::NvEglRenderer(const char *name, uint32_t width, uint32_t height, uint32_t x_offset, uint32_t y_offset, const char *ttfFilePath,const char *message,float scale,float r,float g,float b,float fontsize,int textPosX, int textPosY,float opacity) // alwaysOnTOp
{
    int depth;
    int screen_num;
    XSetWindowAttributes window_attributes;
    x_window = 0;
    x_display = NULL;
    XColor color, dummy;
    XGCValues gr_values;

    this->mWidth = width;
    this->mHeight = height;
    this->drawBorder = false;

    this->_x_offset = x_offset;
    this->_y_offset = y_offset;


    texture_id = 0;
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

    setFPS(30);

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

    //window_attributes.override_redirect = 1;
    window_attributes.background_pixel =
        BlackPixel(x_display, DefaultScreen(x_display));

    window_attributes.override_redirect = (displayOnTop ? 1 : 0);
    Atom WM_HINTS; 
    if(window_attributes.override_redirect == 0)
    { 
       struct
       {
          unsigned long flags;
          unsigned long functions;
          unsigned long decorations;
          long inputMode;
          unsigned long status;
       } WM_HINTS = { (1L << 1), 0, 0, 0, 0 }; // this we added to remove title bar
    }

    x_window = XCreateWindow(x_display,
                             DefaultRootWindow(x_display), x_offset,
                             y_offset, width, height,
                             0,
                             depth, CopyFromParent,
                             CopyFromParent,
                             (CWBackPixel | CWOverrideRedirect),
                             &window_attributes);

    if(window_attributes.override_redirect == 0)
    {
       XStoreName(x_display, x_window, "ApraEglRenderer");
       XFlush(x_display);
    
       XSizeHints hints;
       hints.x = x_offset;
       hints.y = y_offset;
       hints.width = width;
       hints.height = height;
       hints.flags = PPosition | PSize;
       XSetWMNormalHints(x_display, x_window, &hints);

       WM_HINTS = XInternAtom(x_display, "_MOTIF_WM_HINTS", True);
       XChangeProperty(x_display, x_window, WM_HINTS, WM_HINTS, 32,
                PropModeReplace, (unsigned char *)&WM_HINTS, 5);
    }

    XSelectInput(x_display, (int32_t) x_window, ButtonPressMask |
								                NoEventMask |
								                KeyPressMask |
								                KeyReleaseMask |
								                ButtonReleaseMask |
								                EnterWindowMask |
								                LeaveWindowMask |
								                PointerMotionMask |
								                PointerMotionHintMask |
								                Button1MotionMask |
								                Button2MotionMask |
								                Button3MotionMask |
								                Button4MotionMask |
								                Button5MotionMask |
								                ButtonMotionMask |
								                KeymapStateMask |
								                ExposureMask |
								                VisibilityChangeMask |
								                StructureNotifyMask |
								                ResizeRedirectMask |
								                SubstructureNotifyMask |
								                SubstructureRedirectMask |
								                FocusChangeMask |
								                PropertyChangeMask |
								                ColormapChangeMask |
								                OwnerGrabButtonMask);

        fontinfo = XLoadQueryFont(x_display, "9x15bold");

        // XAllocNamedColor(x_display, DefaultColormap(x_display, screen_num), "green", &color, &dummy);
        // XSetWindowBorder(x_display, x_window, color.pixel);

        // gr_values.font = fontinfo->fid;
        // gr_values.foreground = color.pixel;
        // gr_values.line_width = 5;

        // gc = XCreateGC(x_display, x_window, GCFont | GCForeground | GCLineWidth, &gr_values);

        // XFlush(x_display);
        // XMapWindow(x_display, (int32_t)x_window);
        // XFlush(x_display);

        XMapWindow(x_display, (int32_t)x_window);
        gc = XCreateGC(x_display, x_window, 0, NULL);

    XSetForeground(x_display, gc,
                WhitePixel(x_display, DefaultScreen(x_display)) );
    fontinfo = XLoadQueryFont(x_display, "9x15bold");

    pthread_mutex_lock(&render_lock);
    pthread_create(&render_thread, NULL, renderThread, this);
    pthread_setname_np(render_thread, "EglRenderer");
    pthread_cond_wait(&render_cond, &render_lock);
    pthread_mutex_unlock(&render_lock);
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
    setWindowOpacity(opacity);
    return;
}

bool NvEglRenderer::renderAndDrawLoop()
{
    if (drawBorder)
    {
        XDrawRectangle(x_display, x_window, gc, 0, 0, (mWidth)-1, (mHeight)-1);
        XFlush(x_display);
    }
    return true;
}

bool NvEglRenderer::windowDrag()
{
    if (XCheckMaskEvent(x_display,
                        ButtonPressMask |
                            NoEventMask |
                            KeyPressMask |
                            KeyReleaseMask |
                            ButtonReleaseMask |
                            EnterWindowMask |
                            LeaveWindowMask |
                            PointerMotionMask |
                            PointerMotionHintMask |
                            Button1MotionMask |
                            Button2MotionMask |
                            Button3MotionMask |
                            Button4MotionMask |
                            Button5MotionMask |
                            ButtonMotionMask |
                            KeymapStateMask |
                            ExposureMask |
                            VisibilityChangeMask |
                            StructureNotifyMask |
                            ResizeRedirectMask |
                            SubstructureNotifyMask |
                            SubstructureRedirectMask |
                            FocusChangeMask |
                            PropertyChangeMask |
                            ColormapChangeMask |
                            OwnerGrabButtonMask,
                        &event))
    {
        if (event.type == ButtonPress)
        {
            if (event.xbutton.button == Button1)
            {
                drag_start_x = event.xbutton.x_root - _x_offset;
                drag_start_y = event.xbutton.y_root - _y_offset;
                is_dragging = true;
            }
        }
        else if (event.type == MotionNotify)
        {
            if (is_dragging)
            {
                int screen = DefaultScreen(x_display);
                _x_offset = event.xbutton.x_root - drag_start_x;
                _y_offset = event.xbutton.y_root - drag_start_y;
                int centerX = _x_offset + mWidth / 2;
                int centerY = _y_offset + mHeight / 2;
                int screenWidth = XDisplayWidth(x_display, screen);
                int screenHeight = XDisplayHeight(x_display, screen);

                // Determine the closest corner
                int closestX, closestY;

                if (centerX <= screenWidth / 2)
                {
                    closestX = 0;
                }
                else
                {
                    closestX = screenWidth - mWidth;
                }

                if (centerY <= screenHeight / 2)
                {
                    closestY = 0;
                }
                else
                {
                    closestY = screenHeight - mHeight;
                }

                // Move the window to the closest corner
                // XMoveWindow(x_display, x_window, _x_offset, _y_offset);
                XMoveWindow(x_display, x_window, closestX, closestY);
                XFlush(x_display);
            }
        }
        else if (event.type == ButtonRelease)
        {
            if (event.xbutton.button == Button1)
            {
                is_dragging = false;
            }
        }
    }
    return true;
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

        renderer->windowDrag();
        renderer->renderInternal();
        renderer->renderAndDrawLoop();
        pthread_mutex_lock(&renderer->render_lock);
        pthread_cond_broadcast(&renderer->render_cond);
    }
    pthread_mutex_unlock(&renderer->render_lock);

finish:
    if (renderer->texture_id)
    {
        glDeleteTextures(1, &renderer->texture_id);
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

int
NvEglRenderer::renderInternal()
{
    EGLImageKHR hEglImage;
    bool frame_is_late = false;

    EGLSyncKHR egl_sync;
    int iErr;

    // Import dmabuf using explicit attributes (supports up to 3 planes)
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
    hEglImage = eglCreateImage(
        egl_display,
        EGL_NO_CONTEXT,
        EGL_LINUX_DMA_BUF_EXT,
        (EGLClientBuffer)NULL,
        attrs
    );
    if (!hEglImage)
    {
        return -1;
    }

    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_EXTERNAL_OES, texture_id);
    glTexParameteri(GL_TEXTURE_EXTERNAL_OES, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_EXTERNAL_OES, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_EXTERNAL_OES, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_EXTERNAL_OES, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glEGLImageTargetTexture2DOES(GL_TEXTURE_EXTERNAL_OES, hEglImage);
    glDrawArrays(GL_TRIANGLES, 0, 6);
    EGLBoolean egl_status = eglGetError();
    if (egl_status != EGL_SUCCESS)
    {
        eglDestroyImageKHR(egl_display, hEglImage);
        return -1;
    }

    iErr = glGetError();
    if (iErr != GL_NO_ERROR)
    {
        return -1;
    }
    egl_sync = eglCreateSyncKHR(egl_display, EGL_SYNC_FENCE_KHR, NULL);
    if (egl_sync == EGL_NO_SYNC_KHR)
    {
        return -1;
    }
    if (last_render_time.tv_sec != 0)
    {
        pthread_mutex_lock(&render_lock);
        last_render_time.tv_sec += render_time_sec;
        last_render_time.tv_nsec += render_time_nsec;
        last_render_time.tv_sec += last_render_time.tv_nsec / 1000000000UL;
        last_render_time.tv_nsec %= 1000000000UL;

        pthread_cond_timedwait(&render_cond, &render_lock,
                &last_render_time);

        pthread_mutex_unlock(&render_lock);
    }
    else
    {
        struct timeval now;

        gettimeofday(&now, NULL);
        last_render_time.tv_sec = now.tv_sec;
        last_render_time.tv_nsec = now.tv_usec * 1000L;
    }

    // Ensure video program is bound before drawing the frame
    glUseProgram(gl_program);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_EXTERNAL_OES, texture_id);
    glDrawArrays(GL_TRIANGLES, 0, 6);

    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    GLfloat projection[16] = {
        2.0f / render_width, 0.0f, 0.0f, 0.0f,
        0.0f, -2.0f / render_height, 0.0f, 0.0f,
        0.0f, 0.0f, -1.0f, 0.0f,
    -1.0f, 1.0f, 0.0f, 1.0f
    };

    // Save previous GL state BEFORE switching to text shader
    GLint prevProgram;
    GLint prevVAO = 0;
    GLint prevTexExternal;
    GLint prevTex2D;
    GLint prevArrayBuffer;
    GLint prevActiveTexUnit;
    GLboolean wasBlendEnabled = glIsEnabled(GL_BLEND);
    glGetIntegerv(GL_CURRENT_PROGRAM, &prevProgram);
    if (glGetError() == GL_NO_ERROR && glBindVertexArrayOES)
    {
        glGetIntegerv(GL_VERTEX_ARRAY_BINDING_OES, &prevVAO);
    }
    glGetIntegerv(GL_ACTIVE_TEXTURE, &prevActiveTexUnit);
    glGetIntegerv(GL_TEXTURE_BINDING_EXTERNAL_OES, &prevTexExternal);
    glGetIntegerv(GL_TEXTURE_BINDING_2D, &prevTex2D);
    glGetIntegerv(GL_ARRAY_BUFFER_BINDING, &prevArrayBuffer);

    glUseProgram(textShader);
    GLint projLoc = glGetUniformLocation(textShader, "projection");
    glUniformMatrix4fv(projLoc, 1, GL_FALSE, projection);
    initFontAtlas(ttfFilePath.c_str(), fontSize);

    // Compute multiline-aware stencil rectangle: max line width and total height
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
        Character ch = Characters[*it];
        currentLineW += static_cast<int>((ch.Advance >> 6) * scale);
    }
    if (currentLineW > maxLineW) maxLineW = currentLineW;
    int textW = maxLineW > 0 ? maxLineW : 1;
    int textH = (g_fontLineAdvance > 0 ? g_fontLineAdvance : 16) * numLines * scale;
    if (textH < 1) textH = 1;

    // Save previous scissor state
    GLboolean prevScissorEnabled = glIsEnabled(GL_SCISSOR_TEST);
    GLint prevScissorBox[4];
    glGetIntegerv(GL_SCISSOR_BOX, prevScissorBox);

    // Prepare stencil: 0 everywhere, 1 inside text rectangle
    glEnable(GL_STENCIL_TEST);
    glStencilMask(0xFF);
    glDisable(GL_SCISSOR_TEST);
    glClearStencil(0);
    glClear(GL_STENCIL_BUFFER_BIT);

    // Compute scissor box for top-left coordinates to GL bottom-left origin
    // Account for font ascent (text can extend above baseline)
    int ascentPixels = g_fontAscent > 0 ? static_cast<int>(g_fontAscent * scale) : static_cast<int>(16 * scale);
    int adjustedtextPosY = static_cast<int>(textPosY) - ascentPixels;
    int adjustedTextH = textH + ascentPixels;
    if (adjustedtextPosY < 0) {
        adjustedTextH += adjustedtextPosY; // Reduce height if we go negative
        adjustedtextPosY = 0;
    }
    
    GLint scX = static_cast<GLint>(textPosX);
    GLint scY = static_cast<GLint>(static_cast<int>(render_height) - adjustedtextPosY - adjustedTextH);
    if (scY < 0) scY = 0;
    if (scX < 0) scX = 0;
    GLsizei scW = static_cast<GLsizei>(textW);
    GLsizei scH = static_cast<GLsizei>(adjustedTextH);

    glEnable(GL_SCISSOR_TEST);
    glScissor(scX, scY, scW, scH);
    glClearStencil(1);
    glClear(GL_STENCIL_BUFFER_BIT);

    // Only pass where stencil == 1
    glStencilFunc(GL_EQUAL, 1, 0xFF);
    glStencilOp(GL_KEEP, GL_KEEP, GL_KEEP);

    // For non-VAO path, set attrib pointer for text
    if (!glBindVertexArrayOES)
    {
        glBindBuffer(GL_ARRAY_BUFFER, textVBO);
        glEnableVertexAttribArray(1);
        glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, 4 * sizeof(float), 0);
    }

    // --- render text ---
    RenderText(message, textPosX, textPosY, scale, r,g,b);

    // Restore previously saved GL state
    glUseProgram(prevProgram);
    if (glBindVertexArrayOES && prevVAO)
    {
        glBindVertexArrayOES(prevVAO);
    }
    glActiveTexture(prevActiveTexUnit);
    glBindTexture(GL_TEXTURE_2D, prevTex2D);
    glBindTexture(GL_TEXTURE_EXTERNAL_OES, prevTexExternal);
    glBindBuffer(GL_ARRAY_BUFFER, prevArrayBuffer);
    // Disable stencil and restore scissor state
    glDisable(GL_STENCIL_TEST);
    if (prevScissorEnabled)
    {
        glEnable(GL_SCISSOR_TEST);
        glScissor(prevScissorBox[0], prevScissorBox[1], prevScissorBox[2], prevScissorBox[3]);
    }
    else
    {
        glDisable(GL_SCISSOR_TEST);
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
    if (!wasBlendEnabled)
    {
        glDisable(GL_BLEND);
    }

    eglSwapBuffers(egl_display, egl_surface);

    if (eglGetError() != EGL_SUCCESS)
    {
        return -1;
    }
    if (eglClientWaitSyncKHR (egl_display, egl_sync,
                EGL_SYNC_FLUSH_COMMANDS_BIT_KHR, EGL_FOREVER_KHR) == EGL_FALSE)
    {
    }

    if (eglDestroySyncKHR(egl_display, egl_sync) != EGL_TRUE)
    {
    }
    eglDestroyImageKHR(egl_display, hEglImage);

    if (strlen(overlay_str) != 0)
    {
        XSetForeground(x_display, gc,
                        BlackPixel(x_display, DefaultScreen(x_display)));
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
                               const char *message, float scale, float r, float g, float b, float fontsize,int textPosX, int textPosY,float opacity)
{
    NvEglRenderer* renderer = new NvEglRenderer(name, width, height,
                                    x_offset, y_offset, ttfFilePath, message, scale, r, g, b, fontsize, textPosX, textPosY, opacity);
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
        "precision mediump float;\n" "varying vec2 interp_tc; \n"
        "uniform samplerExternalOES tex; \n" "void main() {\n"
        "gl_FragColor = texture2D(tex, interp_tc);\n" "}\n";

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

    this->alpha_location = -1;
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
    void main() {
        float alpha = texture2D(text, TexCoords).a;
        gl_FragColor = vec4(textColor, alpha);
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
    // Bind text vertex attribute to location 1 to avoid clobbering video attrib 0
    glBindAttribLocation(shader, 1, "vertex");
    glLinkProgram(shader);

    // Bind sampler uniform to texture unit 0
    glUseProgram(shader);
    GLint textSamplerLoc = glGetUniformLocation(shader, "text");
    if (textSamplerLoc != -1)
    {
        glUniform1i(textSamplerLoc, 0);
    }
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

        Character ch = Characters[*c];

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
