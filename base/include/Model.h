void model_init (void);
void model_draw (void);
void draw_frames(void);
void drawCameraFrame(void* frameData, int width, int height);
const float *model_matrix(void);
void model_pan_start (int x, int y);
void model_pan_move (int x, int y);
