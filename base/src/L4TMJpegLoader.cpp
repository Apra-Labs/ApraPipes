#include "L4TMJpegLoader.h"
#include "Logger.h"
#include <dlfcn.h>
#include <mutex>

/**
 * Dynamic loader for NVIDIA's libnvjpeg.so
 *
 * This module resolves the libjpeg version conflict between vcpkg (v62) and
 * NVIDIA's L4T Multimedia API (v80) by dynamically loading functions from
 * libnvjpeg.so at runtime, bypassing the static linker.
 *
 * The struct sizes differ significantly:
 *   - vcpkg:  jpeg_compress_struct=520, jpeg_decompress_struct=632
 *   - NVIDIA: jpeg_compress_struct=728, jpeg_decompress_struct=776
 *
 * L4TM code compiled with NVIDIA headers uses the larger structs, so we must
 * call libnvjpeg.so's implementation which expects those struct sizes.
 */

namespace L4TMJpegLoader {

namespace {

// Handle to libnvjpeg.so
void* g_handle = nullptr;
bool g_initialized = false;
bool g_available = false;
std::once_flag g_initFlag;

// Function pointer types
using jpeg_std_error_t = struct jpeg_error_mgr* (*)(struct jpeg_error_mgr*);
using jpeg_CreateCompress_t = void (*)(j_compress_ptr, int, size_t);
using jpeg_CreateDecompress_t = void (*)(j_decompress_ptr, int, size_t);
using jpeg_destroy_compress_t = void (*)(j_compress_ptr);
using jpeg_destroy_decompress_t = void (*)(j_decompress_ptr);
using jpeg_mem_dest_t = void (*)(j_compress_ptr, unsigned char**, unsigned long*);
using jpeg_mem_src_t = void (*)(j_decompress_ptr, unsigned char*, unsigned long);
using jpeg_set_defaults_t = void (*)(j_compress_ptr);
using jpeg_set_quality_t = void (*)(j_compress_ptr, int, boolean);
using jpeg_suppress_tables_t = void (*)(j_compress_ptr, boolean);
using jpeg_start_compress_t = void (*)(j_compress_ptr, boolean);
using jpeg_write_scanlines_t = JDIMENSION (*)(j_compress_ptr, JSAMPARRAY, JDIMENSION);
using jpeg_write_raw_data_t = JDIMENSION (*)(j_compress_ptr, JSAMPIMAGE, JDIMENSION);
using jpeg_finish_compress_t = void (*)(j_compress_ptr);
using jpeg_read_header_t = int (*)(j_decompress_ptr, boolean);
using jpeg_start_decompress_t = boolean (*)(j_decompress_ptr);
using jpeg_read_raw_data_t = JDIMENSION (*)(j_decompress_ptr, JSAMPIMAGE, JDIMENSION);
using jpeg_finish_decompress_t = boolean (*)(j_decompress_ptr);
using jpeg_set_hardware_acceleration_parameters_enc_t = void (*)(j_compress_ptr, boolean, unsigned int, unsigned int, unsigned int);

// Function pointers
jpeg_std_error_t fn_jpeg_std_error = nullptr;
jpeg_CreateCompress_t fn_jpeg_CreateCompress = nullptr;
jpeg_CreateDecompress_t fn_jpeg_CreateDecompress = nullptr;
jpeg_destroy_compress_t fn_jpeg_destroy_compress = nullptr;
jpeg_destroy_decompress_t fn_jpeg_destroy_decompress = nullptr;
jpeg_mem_dest_t fn_jpeg_mem_dest = nullptr;
jpeg_mem_src_t fn_jpeg_mem_src = nullptr;
jpeg_set_defaults_t fn_jpeg_set_defaults = nullptr;
jpeg_set_quality_t fn_jpeg_set_quality = nullptr;
jpeg_suppress_tables_t fn_jpeg_suppress_tables = nullptr;
jpeg_start_compress_t fn_jpeg_start_compress = nullptr;
jpeg_write_scanlines_t fn_jpeg_write_scanlines = nullptr;
jpeg_write_raw_data_t fn_jpeg_write_raw_data = nullptr;
jpeg_finish_compress_t fn_jpeg_finish_compress = nullptr;
jpeg_read_header_t fn_jpeg_read_header = nullptr;
jpeg_start_decompress_t fn_jpeg_start_decompress = nullptr;
jpeg_read_raw_data_t fn_jpeg_read_raw_data = nullptr;
jpeg_finish_decompress_t fn_jpeg_finish_decompress = nullptr;
jpeg_set_hardware_acceleration_parameters_enc_t fn_jpeg_set_hardware_acceleration_parameters_enc = nullptr;

template<typename T>
T loadSymbol(const char* name) {
    T fn = reinterpret_cast<T>(dlsym(g_handle, name));
    if (!fn) {
        LOG_ERROR << "L4TMJpegLoader: Failed to load symbol '" << name << "': " << dlerror();
    }
    return fn;
}

void doInit() {
    g_initialized = true;

    // Try to load libnvjpeg.so from the Tegra libs path
    // On Jetson, this is typically at /usr/lib/aarch64-linux-gnu/tegra/libnvjpeg.so
    const char* paths[] = {
        "libnvjpeg.so",
        "/usr/lib/aarch64-linux-gnu/tegra/libnvjpeg.so",
        "/usr/lib/aarch64-linux-gnu/libnvjpeg.so",
        nullptr
    };

    for (int i = 0; paths[i] != nullptr; ++i) {
        g_handle = dlopen(paths[i], RTLD_NOW | RTLD_LOCAL);
        if (g_handle) {
            LOG_INFO << "L4TMJpegLoader: Loaded " << paths[i];
            break;
        }
    }

    if (!g_handle) {
        LOG_ERROR << "L4TMJpegLoader: Failed to load libnvjpeg.so: " << dlerror();
        return;
    }

    // Load all function pointers
    fn_jpeg_std_error = loadSymbol<jpeg_std_error_t>("jpeg_std_error");
    fn_jpeg_CreateCompress = loadSymbol<jpeg_CreateCompress_t>("jpeg_CreateCompress");
    fn_jpeg_CreateDecompress = loadSymbol<jpeg_CreateDecompress_t>("jpeg_CreateDecompress");
    fn_jpeg_destroy_compress = loadSymbol<jpeg_destroy_compress_t>("jpeg_destroy_compress");
    fn_jpeg_destroy_decompress = loadSymbol<jpeg_destroy_decompress_t>("jpeg_destroy_decompress");
    fn_jpeg_mem_dest = loadSymbol<jpeg_mem_dest_t>("jpeg_mem_dest");
    fn_jpeg_mem_src = loadSymbol<jpeg_mem_src_t>("jpeg_mem_src");
    fn_jpeg_set_defaults = loadSymbol<jpeg_set_defaults_t>("jpeg_set_defaults");
    fn_jpeg_set_quality = loadSymbol<jpeg_set_quality_t>("jpeg_set_quality");
    fn_jpeg_suppress_tables = loadSymbol<jpeg_suppress_tables_t>("jpeg_suppress_tables");
    fn_jpeg_start_compress = loadSymbol<jpeg_start_compress_t>("jpeg_start_compress");
    fn_jpeg_write_scanlines = loadSymbol<jpeg_write_scanlines_t>("jpeg_write_scanlines");
    fn_jpeg_write_raw_data = loadSymbol<jpeg_write_raw_data_t>("jpeg_write_raw_data");
    fn_jpeg_finish_compress = loadSymbol<jpeg_finish_compress_t>("jpeg_finish_compress");
    fn_jpeg_read_header = loadSymbol<jpeg_read_header_t>("jpeg_read_header");
    fn_jpeg_start_decompress = loadSymbol<jpeg_start_decompress_t>("jpeg_start_decompress");
    fn_jpeg_read_raw_data = loadSymbol<jpeg_read_raw_data_t>("jpeg_read_raw_data");
    fn_jpeg_finish_decompress = loadSymbol<jpeg_finish_decompress_t>("jpeg_finish_decompress");
    fn_jpeg_set_hardware_acceleration_parameters_enc = loadSymbol<jpeg_set_hardware_acceleration_parameters_enc_t>("jpeg_set_hardware_acceleration_parameters_enc");

    // Check that all essential functions were loaded
    g_available = fn_jpeg_std_error &&
                  fn_jpeg_CreateCompress &&
                  fn_jpeg_CreateDecompress &&
                  fn_jpeg_destroy_compress &&
                  fn_jpeg_destroy_decompress &&
                  fn_jpeg_mem_dest &&
                  fn_jpeg_mem_src &&
                  fn_jpeg_set_defaults &&
                  fn_jpeg_set_quality &&
                  fn_jpeg_start_compress &&
                  fn_jpeg_finish_compress &&
                  fn_jpeg_read_header &&
                  fn_jpeg_start_decompress &&
                  fn_jpeg_finish_decompress;

    if (g_available) {
        LOG_INFO << "L4TMJpegLoader: All required symbols loaded successfully";
    } else {
        LOG_ERROR << "L4TMJpegLoader: Some required symbols failed to load";
    }
}

} // anonymous namespace

bool init() {
    std::call_once(g_initFlag, doInit);
    return g_available;
}

bool isAvailable() {
    return g_available;
}

// === Compress Functions ===

struct jpeg_error_mgr* std_error(struct jpeg_error_mgr* err) {
    if (!fn_jpeg_std_error) {
        LOG_ERROR << "L4TMJpegLoader: jpeg_std_error not loaded";
        return nullptr;
    }
    return fn_jpeg_std_error(err);
}

void create_compress(j_compress_ptr cinfo) {
    if (!fn_jpeg_CreateCompress) {
        LOG_ERROR << "L4TMJpegLoader: jpeg_CreateCompress not loaded";
        return;
    }
    // Pass NVIDIA's JPEG_LIB_VERSION (80) and struct size (728)
    fn_jpeg_CreateCompress(cinfo, JPEG_LIB_VERSION, sizeof(struct jpeg_compress_struct));
}

void destroy_compress(j_compress_ptr cinfo) {
    if (!fn_jpeg_destroy_compress) {
        LOG_ERROR << "L4TMJpegLoader: jpeg_destroy_compress not loaded";
        return;
    }
    fn_jpeg_destroy_compress(cinfo);
}

void mem_dest(j_compress_ptr cinfo, unsigned char** outbuffer, unsigned long* outsize) {
    if (!fn_jpeg_mem_dest) {
        LOG_ERROR << "L4TMJpegLoader: jpeg_mem_dest not loaded";
        return;
    }
    fn_jpeg_mem_dest(cinfo, outbuffer, outsize);
}

void set_defaults(j_compress_ptr cinfo) {
    if (!fn_jpeg_set_defaults) {
        LOG_ERROR << "L4TMJpegLoader: jpeg_set_defaults not loaded";
        return;
    }
    fn_jpeg_set_defaults(cinfo);
}

void set_quality(j_compress_ptr cinfo, int quality, boolean force_baseline) {
    if (!fn_jpeg_set_quality) {
        LOG_ERROR << "L4TMJpegLoader: jpeg_set_quality not loaded";
        return;
    }
    fn_jpeg_set_quality(cinfo, quality, force_baseline);
}

void suppress_tables(j_compress_ptr cinfo, boolean suppress) {
    if (!fn_jpeg_suppress_tables) {
        LOG_ERROR << "L4TMJpegLoader: jpeg_suppress_tables not loaded";
        return;
    }
    fn_jpeg_suppress_tables(cinfo, suppress);
}

void start_compress(j_compress_ptr cinfo, boolean write_all_tables) {
    if (!fn_jpeg_start_compress) {
        LOG_ERROR << "L4TMJpegLoader: jpeg_start_compress not loaded";
        return;
    }
    fn_jpeg_start_compress(cinfo, write_all_tables);
}

JDIMENSION write_scanlines(j_compress_ptr cinfo, JSAMPARRAY scanlines, JDIMENSION num_lines) {
    if (!fn_jpeg_write_scanlines) {
        LOG_ERROR << "L4TMJpegLoader: jpeg_write_scanlines not loaded";
        return 0;
    }
    return fn_jpeg_write_scanlines(cinfo, scanlines, num_lines);
}

JDIMENSION write_raw_data(j_compress_ptr cinfo, JSAMPIMAGE data, JDIMENSION num_lines) {
    if (!fn_jpeg_write_raw_data) {
        LOG_ERROR << "L4TMJpegLoader: jpeg_write_raw_data not loaded";
        return 0;
    }
    return fn_jpeg_write_raw_data(cinfo, data, num_lines);
}

void finish_compress(j_compress_ptr cinfo) {
    if (!fn_jpeg_finish_compress) {
        LOG_ERROR << "L4TMJpegLoader: jpeg_finish_compress not loaded";
        return;
    }
    fn_jpeg_finish_compress(cinfo);
}

void set_hardware_acceleration_parameters_enc(
    j_compress_ptr cinfo,
    boolean hw_acceleration,
    unsigned int defaultBuffSize,
    unsigned int maxBuffSize,
    unsigned int hwBuffSize) {
    if (!fn_jpeg_set_hardware_acceleration_parameters_enc) {
        LOG_ERROR << "L4TMJpegLoader: jpeg_set_hardware_acceleration_parameters_enc not loaded";
        return;
    }
    fn_jpeg_set_hardware_acceleration_parameters_enc(cinfo, hw_acceleration, defaultBuffSize, maxBuffSize, hwBuffSize);
}

// === Decompress Functions ===

void create_decompress(j_decompress_ptr cinfo) {
    if (!fn_jpeg_CreateDecompress) {
        LOG_ERROR << "L4TMJpegLoader: jpeg_CreateDecompress not loaded";
        return;
    }
    // Pass NVIDIA's JPEG_LIB_VERSION (80) and struct size (776)
    fn_jpeg_CreateDecompress(cinfo, JPEG_LIB_VERSION, sizeof(struct jpeg_decompress_struct));
}

void destroy_decompress(j_decompress_ptr cinfo) {
    if (!fn_jpeg_destroy_decompress) {
        LOG_ERROR << "L4TMJpegLoader: jpeg_destroy_decompress not loaded";
        return;
    }
    fn_jpeg_destroy_decompress(cinfo);
}

void mem_src(j_decompress_ptr cinfo, unsigned char* inbuffer, unsigned long insize) {
    if (!fn_jpeg_mem_src) {
        LOG_ERROR << "L4TMJpegLoader: jpeg_mem_src not loaded";
        return;
    }
    fn_jpeg_mem_src(cinfo, inbuffer, insize);
}

int read_header(j_decompress_ptr cinfo, boolean require_image) {
    if (!fn_jpeg_read_header) {
        LOG_ERROR << "L4TMJpegLoader: jpeg_read_header not loaded";
        return -1;
    }
    return fn_jpeg_read_header(cinfo, require_image);
}

boolean start_decompress(j_decompress_ptr cinfo) {
    if (!fn_jpeg_start_decompress) {
        LOG_ERROR << "L4TMJpegLoader: jpeg_start_decompress not loaded";
        return FALSE;
    }
    return fn_jpeg_start_decompress(cinfo);
}

JDIMENSION read_raw_data(j_decompress_ptr cinfo, JSAMPIMAGE data, JDIMENSION max_lines) {
    if (!fn_jpeg_read_raw_data) {
        LOG_ERROR << "L4TMJpegLoader: jpeg_read_raw_data not loaded";
        return 0;
    }
    return fn_jpeg_read_raw_data(cinfo, data, max_lines);
}

boolean finish_decompress(j_decompress_ptr cinfo) {
    if (!fn_jpeg_finish_decompress) {
        LOG_ERROR << "L4TMJpegLoader: jpeg_finish_decompress not loaded";
        return FALSE;
    }
    return fn_jpeg_finish_decompress(cinfo);
}

} // namespace L4TMJpegLoader
