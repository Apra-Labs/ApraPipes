#include <stdbool.h>
#include <stdlib.h>
#include <GL/gl.h>
#include <gtk/gtk.h>

#include "Logger.h"
#include "GtkGlRenderer.h"
#include "DMAFDWrapper.h"
#include "Background.h"
#include "Matrix.h"
#include "Model.h"
#include "Program.h"
#include "GLUtils.h"
#include "View.h"

static bool gtkApplicationQuit = false;
std::thread t;
struct signal
{
    const gchar *signal;
    GCallback handler;
    GdkEventMask mask;
};

class GtkGlRenderer::Detail
{

public:
    Detail()
    {
    }

    ~Detail()
    {
    }
    static void
    on_resize(GtkGLArea *area, gint width, gint height)
    {
        view_set_window(width, height);
        background_set_window(width, height);
    }

    static gboolean
    on_render(GtkGLArea *glarea, GdkGLContext *context)
    {
        // Clear canvas:
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        // Draw background:
        background_draw();

        // Draw model:
        // model_draw();
        draw_frames();

        // Don't propagate signal:
        return TRUE;
    }

    static void
    on_realize(GtkGLArea *glarea)
    {
        // Make current:
        gtk_gl_area_make_current(glarea);

        // Print version info:
        const GLubyte *renderer = glGetString(GL_RENDERER);
        const GLubyte *version = glGetString(GL_VERSION);

        LOG_INFO << "OpenGL version supported " << version;

        // Enable depth buffer:
        gtk_gl_area_set_has_depth_buffer(glarea, TRUE);

        // Init programs:
        programs_init();

        // Init background:
        background_init();

        // Init model:
        model_init();

        // Get frame clock:
        GdkGLContext *glcontext = gtk_gl_area_get_context(glarea);
        GdkWindow *glwindow = gdk_gl_context_get_window(glcontext);
        GdkFrameClock *frame_clock = gdk_window_get_frame_clock(glwindow);

        // Connect update signal:
        g_signal_connect_swapped(frame_clock, "update", G_CALLBACK(gtk_gl_area_queue_render), glarea);

        // Start updating:
        gdk_frame_clock_begin_updating(frame_clock);
    }

    // static gboolean
    // on_button_press(GtkWidget *widget, GdkEventButton *event)
    // {
    //     GtkAllocation allocation;
    //     gtk_widget_get_allocation(widget, &allocation);

    //     if (event->button == 1)
    //         if (panning == FALSE)
    //         {
    //             panning = TRUE;
    //             model_pan_start(event->x, allocation.height - event->y);
    //         }

    //     return FALSE;
    // }

    // static gboolean
    // on_button_release(GtkWidget *widget, GdkEventButton *event)
    // {
    //     if (event->button == 1)
    //         panning = FALSE;

    //     return FALSE;
    // }

    // static gboolean
    // on_motion_notify(GtkWidget *widget, GdkEventMotion *event)
    // {
    //     GtkAllocation allocation;
    //     gtk_widget_get_allocation(widget, &allocation);

    //     if (panning == TRUE)
    //         model_pan_move(event->x, allocation.height - event->y);

    //     return FALSE;
    // }

    static gboolean
    on_scroll(GtkWidget *widget, GdkEventScroll *event)
    {
        switch (event->direction)
        {
        case GDK_SCROLL_UP:
            view_z_decrease();
            break;

        case GDK_SCROLL_DOWN:
            view_z_increase();
            break;

        default:
            break;
        }

        return FALSE;
    }

    static void
    connect_signals(GtkWidget *widget, struct signal *signals, size_t members)
    {
        FOREACH_NELEM(signals, members, s)
        {
            gtk_widget_add_events(widget, s->mask);
            g_signal_connect(widget, s->signal, s->handler, NULL);
        }
    }

    static void
    connect_window_signals(GtkWidget *window)
    {
        struct signal signals[] = {
            {"destroy", G_CALLBACK(gtk_main_quit), (GdkEventMask)0},
        };

        connect_signals(window, signals, NELEM(signals));
    }

    static void
    connect_glarea_signals(GtkWidget *glarea)
    {
        struct signal signals[] = {
            {"realize", G_CALLBACK(on_realize), (GdkEventMask)0},
            {"render", G_CALLBACK(on_render), (GdkEventMask)0},
            {"resize", G_CALLBACK(on_resize), (GdkEventMask)0},
            {"scroll-event", G_CALLBACK(on_scroll), GDK_SCROLL_MASK},
            // {"button-press-event", G_CALLBACK(on_button_press), GDK_BUTTON_PRESS_MASK},
            // {"button-release-event", G_CALLBACK(on_button_release), GDK_BUTTON_RELEASE_MASK},
            // {"motion-notify-event", G_CALLBACK(on_motion_notify), GDK_BUTTON1_MOTION_MASK},
        };

        connect_signals(glarea, signals, NELEM(signals));
    }
    static void gtkMainThreadLaunch()
    {
        // gtk_main();
        gtkApplicationQuit = true;
    }

    static void closeGtkApplication()
    {
        // gtk_main_quit();
    }

    void screenChanged(GtkWidget *widget, GdkScreen *old_screen,
                       gpointer userdata)
    {
        /* To check if the display supports alpha channels, get the visual */
        GdkScreen *screen = gtk_widget_get_screen(widget);
        GdkVisual *visual = gdk_screen_get_rgba_visual(screen);

        if (!visual)
        {
            printf("Your screen does not support alpha channels!\n");
            visual = gdk_screen_get_system_visual(screen);
        }
        else
        {
            printf("Your screen supports alpha channels!\n");
        }

        gtk_widget_set_visual(widget, visual);
    }

    bool init()
    {
        if (!gtk_init_check(NULL, NULL)) // yash argc argv
        {
            fputs("Could not initialize GTK", stderr);
            return false;
        }

        // Create toplevel window, add GtkGLArea:
        m_builder = gtk_builder_new();
        if (!m_builder)
        {
            LOG_ERROR << "Builder not found";
            return false;
        }
        gtk_builder_add_from_file(m_builder, "atlui.glade", NULL);

        window = (GtkWidget *)gtk_window_new(GTK_WINDOW_TOPLEVEL);
        g_object_ref(window);
        gtk_window_set_default_size(GTK_WINDOW(window), windowWidth, windowHeight);
        gtk_window_set_resizable(GTK_WINDOW(window), FALSE);
        gtk_widget_set_app_paintable(window, TRUE);

        screenChanged(window, NULL, NULL);
        do
        {
            gtk_main_iteration();
        } while (gtk_events_pending());

        mainFixed = (GtkWidget *)gtk_builder_get_object(m_builder, "mainWidget");
        gtk_container_add(GTK_CONTAINER(window), mainFixed);
        glarea = (GtkWidget *)gtk_builder_get_object(m_builder, "glareadraw");

        // Connect GTK signals:
        // connect_window_signals(window);
        // connect_glarea_signals(glarea);

        gtk_widget_show_all(window);
        do
        {
            gtk_main_iteration();
        } while (gtk_events_pending());
        // Enter GTK event loop:
        // gtk_main(); // calling from thread
        // t = std::thread(gtkMainThreadLaunch);
        return true;
    }

    GtkBuilder *m_builder;
    GtkWidget *glarea, *mainFixed, *window;
    // static gboolean panning = FALSE;
    std::string gladeFileName;
    int windowWidth, windowHeight;
    // static bool gtkApplicationQuit = false;
};

GtkGlRenderer::GtkGlRenderer(GtkGlRendererProps props) : Module(SINK, "GtkGlRenderer", props)
{
    mDetail.reset(new Detail());
    mDetail->gladeFileName = props.gladeFileName;
    mDetail->windowWidth = props.windowWidth;
    mDetail->windowHeight = props.windowHeight;
}

GtkGlRenderer::~GtkGlRenderer() {}

bool GtkGlRenderer::init()
{
    if (!Module::init())
    {
        return false;
    }
    if (!mDetail->init())
    {
        LOG_ERROR << "Failed To Initialize GtkGl Area ";
        return false;
    }
    return true;
}

bool GtkGlRenderer::process(frame_container &frames)
{
    auto frame = frames.cbegin()->second;
    if (isFrameEmpty(frame))
    {
        LOG_INFO << "Got Empty Frame";
        return true;
    }
        LOG_ERROR << "Got Frames";
    return true;
}

// Need to check on Mem Type Supported
// Already Checked With CPU , Need to check with
// framemetadata_sp metadata = getFirstInputMetadata();
// FrameMetadata::MemType memType = metadata->getMemType();
// if (memType != FrameMetadata::MemType::DMABUF)
// {
// 	LOG_ERROR << "<" << getId() << ">::validateInputPins input memType is expected to be DMABUF. Actual<" << memType << ">";
// 	return false;
// }

bool GtkGlRenderer::validateInputPins()
{
    if (getNumberOfInputPins() < 1)
    {
        LOG_ERROR << "<" << getId() << ">::validateInputPins size is expected to be 1. Actual<" << getNumberOfInputPins() << ">";
        return false;
    }

    return true;
}

bool GtkGlRenderer::term()
{
    bool res = Module::term();
    if (gtkApplicationQuit == false)
    {
        mDetail->closeGtkApplication();
    }
    else
    {
        t.join();
    }
    return res;
}

bool GtkGlRenderer::processSOS(frame_sp &frame)
{
    auto inputMetadata = frame->getMetadata();
    auto frameType = inputMetadata->getFrameType();
    int width = 0;
    int height = 0;

    switch (frameType)
    {
    case FrameMetadata::FrameType::RAW_IMAGE:
    {
        auto metadata = FrameMetadataFactory::downcast<RawImageMetadata>(inputMetadata);
        if (metadata->getImageType() != ImageMetadata::RGBA || metadata->getImageType() != ImageMetadata::RGB)
        {
            throw AIPException(AIP_FATAL, "Unsupported ImageType, Currently Only RGB and RGBA is supported<" + std::to_string(frameType) + ">");
        }
        // renderWidth = metadata->getWidth();
        // renderHeight = metadata->getHeight();
    }
    break;
    case FrameMetadata::FrameType::RAW_IMAGE_PLANAR:
    {
        auto metadata = FrameMetadataFactory::downcast<RawImagePlanarMetadata>(inputMetadata);
        if (metadata->getImageType() != ImageMetadata::RGBA || metadata->getImageType() != ImageMetadata::RGB)
        {
            throw AIPException(AIP_FATAL, "Unsupported ImageType, Currently Only RGB and RGBA is supported<" + std::to_string(frameType) + ">");
        }
        // renderWidth = metadata->getWidth(0);
        // renderHeight = metadata->getHeight(0);
    }
    break;
    default:
        throw AIPException(AIP_FATAL, "Unsupported FrameType<" + std::to_string(frameType) + ">");
    }

    // mDetail->init(renderHeight, renderWidth);
    return true;
}
