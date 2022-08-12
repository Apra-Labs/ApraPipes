/**
 * Copyright (c) 2018 Parrot Drones SAS
 * Copyright (c) 2016 Aurelien Barre
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above copyright
 *     notice, this list of conditions and the following disclaimer in the
 *     documentation and/or other materials provided with the distribution.
 *   * Neither the name of the copyright holders nor the names of its
 *     contributors may be used to endorse or promote products derived from
 *     this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#ifndef _LIBMP4_H_
#define _LIBMP4_H_

#include <inttypes.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

/* To be used for all public API */
#ifdef MP4_API_EXPORTS
#	ifdef _WIN32
#		define MP4_API __declspec(dllexport)
#	else /* !_WIN32 */
#		define MP4_API __attribute__((visibility("default")))
#	endif /* !_WIN32 */
#else /* !MP4_API_EXPORTS */
#	define MP4_API
#endif /* !MP4_API_EXPORTS */


/**
 * MP4 Metadata keys for muxer.
 * Setting the META key also sets the associated UDTA key to the same value,
 * unless previously set.
 * Setting the UDTA key also sets the associated META key to the same value,
 * unless previously set.
 */
#define MP4_META_KEY_FRIENDLY_NAME "com.apple.quicktime.artist"
#define MP4_UDTA_KEY_FRIENDLY_NAME "\251ART"
#define MP4_META_KEY_TITLE "com.apple.quicktime.title"
#define MP4_UDTA_KEY_TITLE "\251nam"
#define MP4_META_KEY_COMMENT "com.apple.quicktime.comment"
#define MP4_UDTA_KEY_COMMENT "\251cmt"
#define MP4_META_KEY_COPYRIGHT "com.apple.quicktime.copyright"
#define MP4_UDTA_KEY_COPYRIGHT "\251cpy"
#define MP4_META_KEY_MEDIA_DATE "com.apple.quicktime.creationdate"
#define MP4_UDTA_KEY_MEDIA_DATE "\251day"
#define MP4_META_KEY_LOCATION "com.apple.quicktime.location.ISO6709"
#define MP4_UDTA_KEY_LOCATION "\251xyz"
#define MP4_META_KEY_MAKER "com.apple.quicktime.make"
#define MP4_UDTA_KEY_MAKER "\251mak"
#define MP4_META_KEY_MODEL "com.apple.quicktime.model"
#define MP4_UDTA_KEY_MODEL "\251mod"
#define MP4_META_KEY_SOFTWARE_VERSION "com.apple.quicktime.software"
#define MP4_UDTA_KEY_SOFTWARE_VERSION "\251swr"


enum mp4_track_type {
	MP4_TRACK_TYPE_UNKNOWN = 0,
	MP4_TRACK_TYPE_VIDEO,
	MP4_TRACK_TYPE_AUDIO,
	MP4_TRACK_TYPE_HINT,
	MP4_TRACK_TYPE_METADATA,
	MP4_TRACK_TYPE_TEXT,
	MP4_TRACK_TYPE_CHAPTERS,
};


enum mp4_video_codec {
	MP4_VIDEO_CODEC_UNKNOWN = 0,
	MP4_VIDEO_CODEC_AVC,
	MP4_VIDEO_CODEC_HEVC,
	MP4_VIDEO_CODEC_MP4V
};


enum mp4_audio_codec {
	MP4_AUDIO_CODEC_UNKNOWN = 0,
	MP4_AUDIO_CODEC_AAC_LC,
};


enum mp4_metadata_cover_type {
	MP4_METADATA_COVER_TYPE_UNKNOWN = 0,
	MP4_METADATA_COVER_TYPE_JPEG,
	MP4_METADATA_COVER_TYPE_PNG,
	MP4_METADATA_COVER_TYPE_BMP,
};


enum mp4_seek_method {
	MP4_SEEK_METHOD_PREVIOUS = 0,
	MP4_SEEK_METHOD_PREVIOUS_SYNC,
	MP4_SEEK_METHOD_NEXT_SYNC,
	MP4_SEEK_METHOD_NEAREST_SYNC,
};


struct mp4_media_info {
	uint64_t duration;
	uint64_t creation_time;
	uint64_t modification_time;
	uint32_t track_count;
};


struct mp4_track_info {
	uint32_t id;
	const char *name;
	int enabled;
	int in_movie;
	int in_preview;
	enum mp4_track_type type;
	uint32_t timescale;
	uint64_t duration;
	uint64_t creation_time;
	uint64_t modification_time;
	uint32_t sample_count;
	uint32_t sample_max_size;
	const uint64_t *sample_offsets;
	const uint32_t *sample_sizes;
	enum mp4_video_codec video_codec;
	uint32_t video_width;
	uint32_t video_height;
	enum mp4_audio_codec audio_codec;
	uint32_t audio_channel_count;
	uint32_t audio_sample_size;
	float audio_sample_rate;
	const char *content_encoding;
	const char *mime_format;
	int has_metadata;
	const char *metadata_content_encoding;
	const char *metadata_mime_format;
};


/* hvcC box structure */
struct mp4_hvcc_info {
	uint8_t general_profile_space;
	uint8_t general_tier_flag;
	uint8_t general_profile_idc;
	uint32_t general_profile_compatibility_flags;
	uint64_t general_constraints_indicator_flags;
	uint8_t general_level_idc;
	uint16_t min_spatial_segmentation_idc;
	uint8_t parallelism_type;
	uint8_t chroma_format;
	uint8_t bit_depth_luma;
	uint8_t bit_depth_chroma;
	uint16_t avg_framerate;
	uint8_t constant_framerate;
	uint8_t num_temporal_layers;
	uint8_t temporal_id_nested;
	uint8_t length_size;
};


struct mp4_video_decoder_config {
	enum mp4_video_codec codec;
	union {
		struct {
			union {
				uint8_t *sps;
				const uint8_t *c_sps;
			};
			size_t sps_size;
			union {
				uint8_t *pps;
				const uint8_t *c_pps;
			};
			size_t pps_size;
		} avc;
		struct {
			struct mp4_hvcc_info hvcc_info;
			union {
				uint8_t *vps;
				const uint8_t *c_vps;
			};
			size_t vps_size;
			union {
				uint8_t *sps;
				const uint8_t *c_sps;
			};
			size_t sps_size;
			union {
				uint8_t *pps;
				const uint8_t *c_pps;
			};
			size_t pps_size;
		} hevc;
	};
	uint32_t width;
	uint32_t height;
};


struct mp4_track_sample {
	uint32_t size;
	uint32_t metadata_size;
	int silent;
	int sync;
	uint64_t dts;
	uint64_t next_dts;
	uint64_t prev_sync_dts;
	uint64_t next_sync_dts;
};


struct mp4_mux_track_params {
	/* Track type */
	enum mp4_track_type type;
	/* Track name, if NULL, an empty string will be used */
	const char *name;
	/* Track flags (bool-like) */
	int enabled;
	int in_movie;
	int in_preview;
	/* Track timescale, mandatory */
	uint32_t timescale;
	/* Creation time */
	uint64_t creation_time;
	/* Modification time. If zero, creation time will be used */
	uint64_t modification_time;
};


struct mp4_mux_sample {
	const uint8_t *buffer;
	size_t len;
	int sync;
	int64_t dts;
};


struct mp4_mux_scattered_sample {
	const uint8_t *const *buffers;
	const size_t *len;
	int nbuffers;
	int sync;
	int64_t dts;
};


/* Demuxer API */

struct mp4_demux;


MP4_API int mp4_demux_open(const char *filename, struct mp4_demux **ret_obj);


MP4_API int mp4_demux_close(struct mp4_demux *demux);


MP4_API int mp4_demux_get_media_info(struct mp4_demux *demux,
				     struct mp4_media_info *media_info);


MP4_API int mp4_demux_get_track_count(struct mp4_demux *demux);


MP4_API int mp4_demux_get_track_info(struct mp4_demux *demux,
				     unsigned int track_idx,
				     struct mp4_track_info *track_info);


MP4_API int
mp4_demux_get_track_video_decoder_config(struct mp4_demux *demux,
					 unsigned int track_id,
					 struct mp4_video_decoder_config *vdc);


MP4_API int
mp4_demux_get_track_audio_specific_config(struct mp4_demux *demux,
					  unsigned int track_id,
					  uint8_t **audio_specific_config,
					  unsigned int *asc_size);


MP4_API int mp4_demux_get_track_sample(struct mp4_demux *demux,
				       unsigned int track_id,
				       int advance,
				       uint8_t *sample_buffer,
				       unsigned int sample_buffer_size,
				       uint8_t *metadata_buffer,
				       unsigned int metadata_buffer_size,
				       struct mp4_track_sample *track_sample);


MP4_API int mp4_demux_get_track_prev_sample_time(struct mp4_demux *demux,
						 unsigned int track_id,
						 uint64_t *sample_time);


MP4_API int mp4_demux_get_track_next_sample_time(struct mp4_demux *demux,
						 unsigned int track_id,
						 uint64_t *sample_time);


MP4_API int mp4_demux_get_track_prev_sample_time_before(struct mp4_demux *demux,
							unsigned int track_id,
							uint64_t time,
							int sync,
							uint64_t *sample_time);


MP4_API int mp4_demux_get_track_next_sample_time_after(struct mp4_demux *demux,
						       unsigned int track_id,
						       uint64_t time,
						       int sync,
						       uint64_t *sample_time);


MP4_API int mp4_demux_seek(struct mp4_demux *demux,
			   uint64_t time_offset,
			   enum mp4_seek_method method, 
			   int *seekedToFrame);


MP4_API int mp4_demux_seek_to_track_prev_sample(struct mp4_demux *demux,
						unsigned int track_id);


MP4_API int mp4_demux_seek_to_track_next_sample(struct mp4_demux *demux,
						unsigned int track_id);


MP4_API int mp4_demux_get_chapters(struct mp4_demux *demux,
				   unsigned int *chapters_count,
				   uint64_t **chapters_time,
				   char ***chapters_name);


MP4_API int mp4_demux_get_metadata_strings(struct mp4_demux *demux,
					   unsigned int *count,
					   char ***keys,
					   char ***values);


MP4_API int mp4_demux_get_track_metadata_strings(struct mp4_demux *demux,
						 unsigned int track_id,
						 unsigned int *count,
						 char ***keys,
						 char ***values);


MP4_API int
mp4_demux_get_metadata_cover(struct mp4_demux *demux,
			     uint8_t *cover_buffer,
			     unsigned int cover_buffer_size,
			     unsigned int *cover_size,
			     enum mp4_metadata_cover_type *cover_type);


/* Muxer API */

struct mp4_mux;

MP4_API int mp4_mux_open(const char *filename,
			 uint32_t timescale,
			 uint64_t creation_time,
			 uint64_t modification_time,
			 struct mp4_mux **ret_obj);

MP4_API int mp4_mux_open2(const char *filename,
			  uint32_t timescale,
			  uint64_t creation_time,
			  uint64_t modification_time,
			  uint32_t table_size_mbytes,
			  struct mp4_mux **ret_obj);


MP4_API int mp4_mux_sync(struct mp4_mux *mux);


MP4_API int mp4_mux_close(struct mp4_mux *mux);


MP4_API int mp4_mux_add_track(struct mp4_mux *mux,
			      const struct mp4_mux_track_params *params);


MP4_API int mp4_mux_add_ref_to_track(struct mp4_mux *mux,
				     uint32_t track_id,
				     uint32_t ref_track_id);


MP4_API int
mp4_mux_track_set_video_decoder_config(struct mp4_mux *mux,
				       int track_id,
				       struct mp4_video_decoder_config *vdc);


MP4_API int mp4_mux_track_set_audio_specific_config(struct mp4_mux *mux,
						    int track_id,
						    const uint8_t *asc,
						    size_t asc_size,
						    uint32_t channel_count,
						    uint32_t sample_size,
						    float sample_rate);


MP4_API int mp4_mux_track_set_metadata_mime_type(struct mp4_mux *mux,
						 int track_id,
						 const char *content_encoding,
						 const char *mime_type);


MP4_API int mp4_mux_add_file_metadata(struct mp4_mux *mux,
				      const char *key,
				      const char *value);


MP4_API int mp4_mux_add_track_metadata(struct mp4_mux *mux,
				       uint32_t track_id,
				       const char *key,
				       const char *value);


MP4_API int mp4_mux_set_file_cover(struct mp4_mux *mux,
				   enum mp4_metadata_cover_type cover_type,
				   const uint8_t *cover,
				   size_t cover_size);


MP4_API int mp4_mux_track_add_sample(struct mp4_mux *mux,
				     int track_id,
				     const struct mp4_mux_sample *sample);


MP4_API int mp4_mux_track_add_scattered_sample(
	struct mp4_mux *mux,
	int track_id,
	const struct mp4_mux_scattered_sample *sample);


MP4_API void mp4_mux_dump(struct mp4_mux *mux);


/* Utilities */

MP4_API int mp4_generate_avc_decoder_config(const uint8_t *sps,
					    unsigned int sps_size,
					    const uint8_t *pps,
					    unsigned int pps_size,
					    uint8_t *avcc,
					    unsigned int *avcc_size);


MP4_API const char *mp4_track_type_str(enum mp4_track_type type);


MP4_API const char *mp4_video_codec_str(enum mp4_video_codec codec);


MP4_API const char *mp4_audio_codec_str(enum mp4_audio_codec codec);


MP4_API const char *
mp4_metadata_cover_type_str(enum mp4_metadata_cover_type type);


static inline uint64_t mp4_usec_to_sample_time(uint64_t time,
					       uint32_t timescale)
{
	return (time * timescale + 500000) / 1000000;
}


static inline uint64_t mp4_sample_time_to_usec(uint64_t time,
					       uint32_t timescale)
{
	if (timescale == 0)
		return 0;
	return (time * 1000000 + timescale / 2) / timescale;
}


static inline uint64_t mp4_convert_timescale(uint64_t time,
					     uint32_t src_timescale,
					     uint32_t dest_timescale)
{
	if (src_timescale == dest_timescale)
		return time;
	return (time * dest_timescale + src_timescale / 2) / src_timescale;
}

#ifdef __cplusplus
}
#endif /* __cplusplus */

#endif /* !_LIBMP4_H_ */
