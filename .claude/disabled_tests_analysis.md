# Disabled Tests Analysis

**Branch:** `improve-disabled-tests`
**Total Count:** ~169 disabled tests across ~46 files
**Analysis Date:** 2025-12-28
**Status:** TESTING COMPLETED

---

## Test Results Summary

### PASSING SUITES (TO ENABLE)
These entire test suites pass and should be enabled:

| Suite | Tests | Status |
|-------|-------|--------|
| color_conversion_tests | all 11 | PASS |
| jpegdecodercv_tests | all | PASS |
| filereadermodule_tests | all | PASS |
| filewritermodule_tests | all | PASS |
| mp4readersource_tests | all | PASS |
| logger_tests | all | PASS |
| imagemetadata_tests | all | PASS |
| findexstrategy_tests | all | PASS |
| merge_tests | all | PASS |
| split_tests | all | PASS |
| quepushstrategy_tests | all | PASS |
| framesmuxer_tests | all | PASS |
| pipeline_tests | all | PASS |
| valveModule_tests | all | PASS |
| ordered_file_cache | all | PASS |
| rotatecv_tests | all | PASS |
| brightness_contrast_tests | all | PASS |
| virtual_ptz_tests | all | PASS |
| mp4_seek_tests | all | PASS |
| mp4_reverse_play | all | PASS |
| mp4_dts_strategy | all | PASS |
| mp4_getlivevideots_tests | all | PASS |
| overlaymodule_tests | all | PASS |
| TestSignalGenerator_tests | all | PASS |

### INDIVIDUAL PASSING TESTS
| Suite | Test | Status |
|-------|------|--------|
| module_tests | stop | PASS |
| module_tests | stop_bug | PASS |
| module_tests | pipeline_relay | PASS |
| module_tests | feedbackmodule | PASS |
| cv_memory_leaks_tests | cv_mat_memory_leak_2 | PASS |
| cv_memory_leaks_tests | cv_memory_leak_all | PASS |

### FAILING TESTS (KEEP DISABLED)
| Suite | Test | Reason |
|-------|------|--------|
| module_tests | pause_play_step | SIGABRT crash |
| calchistogramcv_tests | entire suite | NaN values, validateOutputPins bugs |
| archivespacemanager_tests | profile | 1 failure |
| unit_tests | params_test | Requires command line args |

---

## Category Summary (Updated)

| Category | Count | Status |
|----------|-------|--------|
| **PASSING (Enable)** | ~24 suites + 6 individual | TO ENABLE |
| Performance/Profiling | ~20 | Keep disabled |
| Jetson/V4L2/NVARGUS | ~35 | Platform-specific |
| DMA Memory Tests | ~8 | Jetson-specific |
| GUI/Display Tests | ~8 | Headless CI |
| Hardware-specific | ~10 | Require hardware |
| CUDA Required | ~40 | Untested (no CUDA on builder) |
| Genuine bugs | ~5 | Keep disabled, needs fixing |

---

## Categories Detail

### Performance/Profiling Tests (OK TO DISABLE)
Benchmarking tests, not functional tests.

| File | Test |
|------|------|
| ccnppi_tests.cpp:771 | perf |
| rotatenppi_tests.cpp:93 | perf |
| jpegencodernvjpeg_tests.cpp:138 | perf |
| resizenppi_tests.cpp:151 | perf |
| resizenppi_jpegencodernvjpeg_tests.cpp:107 | perf |
| bmpconverter_tests.cpp:48 | perf |
| virtualcamerasink_tests.cpp:18 | perf |
| effectsnppi_tests.cpp:528 | yuv420_1920x1080_performance |
| Imageresizecv_tests.cpp:114,148,182,217 | perf, *_profile |
| ImageEncodeCV_tests.cpp:108,148,196 | *_profile |
| jpegencoderl4tm_tests.cpp | *_perf tests |
| archivespacemanager_tests.cpp:117 | profile |
| effectsnppi_tests.cpp:595 | kernel_test |

### Jetson/V4L2/NVARGUS (Platform-Specific)
| File | Tests |
|------|-------|
| nvarguscamerahelper_tests.cpp | basic, invalid_sensor_mode |
| nvarguscamera_tests.cpp | all |
| nvv4l2camerahelper_test.cpp | all |
| nvv4l2camera_test.cpp | all |
| nvtransform_tests.cpp | all |
| jpegdecoderl4tm_tests.cpp | all |
| jpegencoderl4tm_tests.cpp | all except perf |
| l4tm_dec_enc_1_tests.cpp | sample |
| h264encoderv4l2_tests.cpp | all |
| eglrenderer_test.cpp | all |

### DMA Memory Tests (Jetson-Specific)
| File | Test |
|------|------|
| memtypeconversion_tests.cpp | Dma_* tests |
| affinetransform_tests.cpp | DMABUF_RGBA |
| imageviewermodule_tests.cpp | Dma_Renderer_* |

### GUI/Display Tests (Headless CI)
| File | Test |
|------|------|
| imageviewermodule_tests.cpp | open_close_window, viewer_test |
| gtkglrenderer_tests.cpp | windowInit2, getErrorCallback |
| h264decoder_tests.cpp | mp4reader_decoder_eglrenderer |
| motionvector_extractor_and_overlay_tests.cpp | *_render_* |

### Hardware-Specific
| File | Hardware |
|------|----------|
| webcam_source_tests.cpp | Webcam |
| sound_record_tests.cpp | Microphone |
| rtsp_client_tests.cpp | RTSP server |
| rtsppusher_tests.cpp | RTSP server |

### CUDA Required (Untested)
- cudamemcopy_tests.cpp
- resizenppi_tests.cpp (except perf)
- rotatenppi_tests.cpp (except perf)
- ccnppi_tests.cpp
- memtypeconversion_tests.cpp (Host_to_Device_*)
- jpegencodernvjpeg_tests.cpp
- jpegdecodernvjpeg_tests.cpp
- resizenppi_jpegencodernvjpeg_tests.cpp
- nvjpeg_combo_tests.cpp
- overlaynppi_tests.cpp
- effectsnppi_tests.cpp
- nv_mp4_file_tests.cpp
- h264Encodernvcodec_tests.cpp
- h264decoder_tests.cpp

---

## Changes to Make

### Enable Entire Suites (remove `*disabled()` from suite level)
1. color_conversion_tests.cpp:9
2. jpegdecodercv_tests.cpp:16
3. logger_tests.cpp:8
4. imagemetadata_tests.cpp:7
5. findexstrategy_tests.cpp:7
6. merge_tests.cpp:10
7. split_tests.cpp:10
8. quepushstrategy_tests.cpp:9
9. framesmuxer_tests.cpp:12
10. pipeline_tests.cpp:14
11. valveModule_tests.cpp:15
12. ordered_cache_of_files_tests.cpp:13
13. rotatecv_tests.cpp:13
14. brightness_contrast_tests.cpp:15
15. virtualptz_tests.cpp:16
16. mp4_seek_tests.cpp:17
17. mp4_reverse_play_tests.cpp:17
18. mp4_dts_strategy_tests.cpp:17
19. mp4_getlivevideots_tests.cpp:16
20. overlaymodule_tests.cpp:12
21. testSignalGeneratorSrc_tests.cpp:12
22. filereadermodule_tests.cpp:15
23. filewritermodule_tests.cpp:13
24. mp4readersource_tests.cpp:17

### Enable Individual Tests (remove `*disabled()` from specific tests)
1. module_tests.cpp:1211 - stop
2. module_tests.cpp:1314 - stop_bug
3. module_tests.cpp:1632 - pipeline_relay
4. module_tests.cpp:1751 - feedbackmodule
5. cv_memory_leaks_tests.cpp:18 - cv_mat_memory_leak_2
6. cv_memory_leaks_tests.cpp:52 - cv_memory_leak_all
