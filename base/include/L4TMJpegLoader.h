#ifndef L4TM_JPEG_LOADER_H
#define L4TM_JPEG_LOADER_H

/**
 * L4TMJpegLoader - Dynamic loader for NVIDIA's libnvjpeg.so
 *
 * This wrapper exists to resolve a symbol conflict between:
 * 1. vcpkg's libjpeg-turbo (linked statically, version 62, struct size 520/632)
 * 2. NVIDIA's libnvjpeg.so (contains libjpeg version 80, struct size 728/776)
 *
 * L4TM modules (JPEGEncoderL4TM, JPEGDecoderL4TM) need NVIDIA's extended
 * jpeg structs with fields like image_scale, scaled_image_width, etc.
 *
 * By using dlopen/dlsym, we bypass the linker's symbol resolution and
 * ensure L4TM code calls the correct NVIDIA implementation.
 *
 * See: docs/declarative-pipeline/JETSON_KNOWN_ISSUES.md (Issue J1)
 */

// Required for NVIDIA's jpeglib.h which uses size_t and FILE
#include <cstddef>
#include <cstdio>

#include "libjpeg-8b/jpeglib.h"

namespace L4TMJpegLoader {

/**
 * Initialize the loader. Must be called before any other functions.
 * Thread-safe (uses call_once internally).
 * @return true if libnvjpeg.so was loaded successfully
 */
bool init();

/**
 * Check if libnvjpeg.so is available and loaded.
 * @return true if loaded and ready to use
 */
bool isAvailable();

// === Compress Functions (Encoder) ===

struct jpeg_error_mgr* std_error(struct jpeg_error_mgr* err);
void create_compress(j_compress_ptr cinfo);
void destroy_compress(j_compress_ptr cinfo);
void mem_dest(j_compress_ptr cinfo, unsigned char** outbuffer, unsigned long* outsize);
void set_defaults(j_compress_ptr cinfo);
void set_quality(j_compress_ptr cinfo, int quality, boolean force_baseline);
void suppress_tables(j_compress_ptr cinfo, boolean suppress);
void start_compress(j_compress_ptr cinfo, boolean write_all_tables);
JDIMENSION write_scanlines(j_compress_ptr cinfo, JSAMPARRAY scanlines, JDIMENSION num_lines);
JDIMENSION write_raw_data(j_compress_ptr cinfo, JSAMPIMAGE data, JDIMENSION num_lines);
void finish_compress(j_compress_ptr cinfo);
void set_hardware_acceleration_parameters_enc(
    j_compress_ptr cinfo,
    boolean hw_acceleration,
    unsigned int defaultBuffSize,
    unsigned int maxBuffSize,
    unsigned int hwBuffSize);

// === Decompress Functions (Decoder) ===

void create_decompress(j_decompress_ptr cinfo);
void destroy_decompress(j_decompress_ptr cinfo);
void mem_src(j_decompress_ptr cinfo, unsigned char* inbuffer, unsigned long insize);
int read_header(j_decompress_ptr cinfo, boolean require_image);
boolean start_decompress(j_decompress_ptr cinfo);
JDIMENSION read_raw_data(j_decompress_ptr cinfo, JSAMPIMAGE data, JDIMENSION max_lines);
boolean finish_decompress(j_decompress_ptr cinfo);

} // namespace L4TMJpegLoader

#endif // L4TM_JPEG_LOADER_H
