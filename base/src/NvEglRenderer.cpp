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
#include "nvbuf_utils.h"
#include "NvElement.h"
#include "Logger.h"
#include <cstring>
#include <sys/time.h>

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

using namespace std;

NvEglRenderer::NvEglRenderer(const char *name, uint32_t width, uint32_t height, uint32_t x_offset, uint32_t y_offset , bool displayOnTop)
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
                       WhitePixel(x_display, DefaultScreen(x_display)));
        fontinfo = XLoadQueryFont(x_display, "9x15bold");
        pthread_mutex_lock(&render_lock);
        pthread_create(&render_thread, NULL, renderThread, this);
        pthread_setname_np(render_thread, "EglRenderer");
        pthread_cond_wait(&render_cond, &render_lock);
        pthread_mutex_unlock(&render_lock);

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
    if (eglGetError() != EGL_SUCCESS)
    {
        goto error;
    }

    if (renderer->InitializeShaders() < 0)
    {
        goto error;
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
    hEglImage = NvEGLImageFromFd(egl_display, render_fd);
    if (!hEglImage)
    {
        return -1;
    }

    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_EXTERNAL_OES, texture_id);
    glEGLImageTargetTexture2DOES(GL_TEXTURE_EXTERNAL_OES, hEglImage);
    glDrawArrays(GL_TRIANGLES, 0, 6);

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
    NvDestroyEGLImage(egl_display, hEglImage);

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
                               uint32_t y_offset , bool displayOnTop)
{
    NvEglRenderer* renderer = new NvEglRenderer(name, width, height,
                                    x_offset, y_offset , displayOnTop);
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

    GLuint vbo; // Store vetex and tex coords
    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertexTexBuf), vertexTexBuf, GL_STATIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    pos_location = glGetAttribLocation(program, "in_pos");

    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glVertexAttribPointer(pos_location, 4, GL_FLOAT, GL_FALSE, 0, (void*)0);
    glEnableVertexAttribArray(pos_location);

    glActiveTexture(GL_TEXTURE0);
    glUniform1i(glGetUniformLocation(program, "texSampler"), 0);
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