#include "EffectsKernel.h"

#define CLAMP_1(x) x < 0 ? 0 : (x > 1 ? 1 : x)
#define CLAMP_255(x) x < 0 ? 0 : (x > 255 ? 255 : x)
#define CLAMP_INT8(x) x < -128 ? -128 : (x > 127 ? 127 : x)

// https://en.wikipedia.org/wiki/YUV#Y%E2%80%B2UV420p_(and_Y%E2%80%B2V12_or_YV12)_to_RGB888_conversion

#define YUV_TO_RGB( Y, U, V, R, G, B )											\
    do																			\
    {																			\
		float rTmp = Y + __fmul_rn (1.370705, V);								\
        float gTmp = Y - __fmul_rn (0.698001, V) - __fmul_rn (0.337633, U);     \
        float bTmp = Y + __fmul_rn (1.732446, U);								\
        R = CLAMP_255(rTmp);													\
        G = CLAMP_255(gTmp);													\
        B = CLAMP_255(bTmp);													\
	} while (0)

#define RGB_TO_Y( R, G, B, Y )															 \
    do																					 \
    {																					 \
		int yTmp = __fmul_rn(R, 0.299) + __fmul_rn (0.587, G) +  __fmul_rn (0.114, B);   \
        Y = CLAMP_255(yTmp);															 \
    } while (0)

#define RGB_TO_UV( R, G, B, U, V )														 \
    do																					 \
    {																					 \
		int uTmp = __fmul_rn(B, 0.436) - __fmul_rn (0.289, G) -  __fmul_rn (0.147, R);   \
        int vTmp = __fmul_rn(R, 0.615) - __fmul_rn (0.515, G) -  __fmul_rn (0.1, B);	 \
        U = 128 + (CLAMP_INT8(uTmp));													 \
        V = 128 + (CLAMP_INT8(vTmp));													 \
	} while (0)

#define RGB_TO_HSV(R, G, B, H, S, V) 									\
do  																	\
{ 																		\
	Npp32f nNormalizedR = __fmul_rn(R, 0.003921569F); /* 255.0F*/ 		\
	Npp32f nNormalizedG = __fmul_rn(G, 0.003921569F); 					\
	Npp32f nNormalizedB = __fmul_rn(B, 0.003921569F); 					\
	Npp32f nS; 															\
	Npp32f nH; 															\
	/* Value*/ 															\
	Npp32f nV = fmaxf(nNormalizedR, nNormalizedG); 						\
	nV = fmaxf(nV, nNormalizedB); 										\
	/*Saturation*/ 														\
	Npp32f nTemp = fminf(nNormalizedR, nNormalizedG); 					\
	nTemp = fminf(nTemp, nNormalizedB); 								\
	Npp32f nDivisor = __fsub_rn(nV, nTemp); 							\
	if (nV == 0.0F) /*achromatics case*/ 								\
	{ 																	\
		nS = 0.0F; 														\
		nH = 0.0F; 														\
	} 																	\
	else /*chromatics case*/ 											\
	{ 																	\
		nS = __fdiv_rn(nDivisor, nV); 									\
	} 																	\
	/* Hue:*/ 															\
	Npp32f nCr = __fdiv_rn(__fsub_rn(nV, nNormalizedR), nDivisor); 		\
	Npp32f nCg = __fdiv_rn(__fsub_rn(nV, nNormalizedG), nDivisor); 		\
	Npp32f nCb = __fdiv_rn(__fsub_rn(nV, nNormalizedB), nDivisor); 		\
	if (nNormalizedR == nV) 											\
		nH = nCb - nCg; 												\
	else if (nNormalizedG == nV) 										\
		nH = __fadd_rn(2.0F, __fsub_rn(nCr, nCb)); 						\
	else if (nNormalizedB == nV) 										\
		nH = __fadd_rn(4.0F, __fsub_rn(nCg, nCr)); 						\
	nH = __fmul_rn(nH, 0.166667F); /* 6.0F*/        					\
	if (nH < 0.0F) 														\
		nH = __fadd_rn(nH, 1.0F); 										\
																		\
	nH = __fmul_rn(nH, 255);											\
	nS = __fmul_rn(nS, 255);											\
	nV = __fmul_rn(nV, 255);											\
																		\
	H = static_cast<Npp8u>(nH); 										\
	S = static_cast<Npp8u>(nS); 										\
	V = static_cast<Npp8u>(nV); 										\
	 																	\
} while(0)

#define HSV_TO_RGB(H, S, V, R, G, B) 																		\
do 																											\
{ 																											\
	Npp32f nNormalizedH = __fmul_rn(H, 0.003921569F); /* 255.0F*/ 											\
	Npp32f nNormalizedS = __fmul_rn(S, 0.003921569F); 														\
	Npp32f nNormalizedV = __fmul_rn(V, 0.003921569F); 														\
	Npp32f nR; 																								\
	Npp32f nG; 																								\
	Npp32f nB; 																								\
	if (nNormalizedS == 0.0F) 																				\
	{ 																										\
		nR = nG = nB = nNormalizedV; 																		\
	} 																										\
	else 																									\
	{ 																										\
		if (nNormalizedH == 1.0F) 																			\
			nNormalizedH = 0.0F; 																			\
		else 																								\
		{																									\
			/* 0.1667F*/																					\
			nNormalizedH = __fmul_rn(nNormalizedH, 6.0F);  													\
		}																									\
	} 																										\
	Npp32f nI = floorf(nNormalizedH); 																		\
	Npp32f nF = nNormalizedH - nI; 																			\
	Npp32f nM = __fmul_rn(nNormalizedV, __fsub_rn(1.0F, nNormalizedS)); 									\
	Npp32f nN = __fmul_rn(nNormalizedV, __fsub_rn(1.0F, __fmul_rn(nNormalizedS, nF) )	); 					\
	Npp32f nK = __fmul_rn(nNormalizedV, __fsub_rn(1.0F, __fmul_rn(nNormalizedS, __fsub_rn(1.0F, nF)) ) ); 	\
	if (nI == 0.0F) 																						\
	{ 																										\
		nR = nNormalizedV; nG = nK; nB = nM; 																\
	} 																										\
	else if (nI == 1.0F) 																					\
	{ 																										\
		nR = nN; nG = nNormalizedV; nB = nM; 																\
	} 																										\
	else if (nI == 2.0F) 																					\
	{ 																										\
		nR = nM; nG = nNormalizedV; nB = nK; 																\
	} 																										\
	else if (nI == 3.0F) 																					\
	{ 																										\
		nR = nM; nG = nN; nB = nNormalizedV; 																\
	} 																										\
	else if (nI == 4.0F) 																					\
	{ 																										\
		nR = nK; nG = nM; nB = nNormalizedV; 																\
	} 																										\
	else if (nI == 5.0F) 																					\
	{ 																										\
		nR = nNormalizedV; nG = nM; nB = nN; 																\
	} 																										\
	nR = __fmul_rn(nR, 255.0F);																				\
	nG = __fmul_rn(nG, 255.0F);																				\
	nB = __fmul_rn(nB, 255.0F);																				\
	R = static_cast<Npp8u>(nR); 																			\
	G = static_cast<Npp8u>(nG); 																			\
	B = static_cast<Npp8u>(nB); 																			\
																											\
} while(0)

#define RGBHUESATURATIONADJUST(r, g, b, R, G, B, hue, saturation)   \
do 																	\
{ 			  														\
	if (hue == 0 && saturation == 1)                                \
	{                                                               \
		R = r;                                                      \
		G = g;                                                      \
		B = b;                                                      \
		break;                                                      \
	}                                                               \
	Npp8u H, S, V; 													\
	RGB_TO_HSV(r, g, b, H, S, V); 									\
	Npp32f Hnew = __fadd_rn(H, hue);		 						\
	Npp32f Snew = __fmul_rn(S, saturation);							\
	H = static_cast<Npp8u>(Hnew);									\
	S = CLAMP_255(Snew); 											\
	HSV_TO_RGB(H, Snew, V, R, G, B);								\
} while(0)

#define BRIGHNESS_CONTRAST(input, output, brightness, contrast)		\
do																	\
{																	\
	output = __fadd_rn(__fmul_rn(input, contrast), brightness);		\
	output = CLAMP_255(output);										\
} while(0)															\

#define BRIGHNESS_CONTRAST_RGB(r, g, b, brightness, contrast)		\
do																	\
{																	\
	BRIGHNESS_CONTRAST(r, r, brightness, contrast);					\
	BRIGHNESS_CONTRAST(g, g, brightness, contrast);					\
	BRIGHNESS_CONTRAST(b, b, brightness, contrast);					\
} while(0)


#define YUVEFFECTS_Y(y, u, v, Y, brightness, contrast, hue, saturation) 				\
do 																						\
{ 																						\
	Npp32f r, g, b; 																	\
	YUV_TO_RGB(y, u, v, r, g, b); 														\
	BRIGHNESS_CONTRAST_RGB(r, g, b, brightness, contrast);								\
	Npp8u R, G, B; 																		\
	RGBHUESATURATIONADJUST(r, g, b, R, G, B, hue, saturation); 							\
	RGB_TO_Y(R, G, B, Y); 																\
} while(0)

#define YUVEFFECTS(y, u, v, Y, U, V, brightness, contrast, hue, saturation)				\
do 																						\
{ 																						\
	Npp32f r, g, b; 																	\
	YUV_TO_RGB(y, u, v, r, g, b); 														\
	BRIGHNESS_CONTRAST_RGB(r, g, b, brightness, contrast);								\
	Npp8u R, G, B; 																		\
	RGBHUESATURATIONADJUST(r, g, b, R, G, B, hue, saturation); 							\
	RGB_TO_Y(R, G, B, Y); 																\
	RGB_TO_UV(R, G, B, U, V); 															\
} while (0)

__global__ void yuv420effects(const uchar4* Yold, const Npp8u* Uold, const Npp8u* Vold, uchar4* Y, Npp8u* U, Npp8u* V, Npp32f brightness, Npp32f contrast, Npp32f hue, Npp32f saturation, int width, int height, int step_y, int step_uv)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x >= width || y >= height)
	{
		return;
	}

	int offset = y * step_y + x;
	auto uvOffset = (y >> 1) * (step_uv)+(x << 1);

	int u_value = Uold[uvOffset] - 128;
	int v_value = Vold[uvOffset] - 128;

	if (y % 2 == 0)
	{
		YUVEFFECTS(Yold[offset].x, u_value, v_value, Y[offset].x, U[uvOffset], V[uvOffset], brightness, contrast, hue, saturation);
	}
	else
	{
		YUVEFFECTS_Y(Yold[offset].x, u_value, v_value, Y[offset].x, brightness, contrast, hue, saturation);
	}
	YUVEFFECTS_Y(Yold[offset].y, u_value, v_value, Y[offset].y, brightness, contrast, hue, saturation);

	uvOffset += 1;
	u_value = Uold[uvOffset] - 128;
	v_value = Vold[uvOffset] - 128;
	if (y % 2 == 0)
	{
		YUVEFFECTS(Yold[offset].z, u_value, v_value, Y[offset].z, U[uvOffset], V[uvOffset], brightness, contrast, hue, saturation);
	}
	else
	{
		YUVEFFECTS_Y(Yold[offset].z, u_value, v_value, Y[offset].z, brightness, contrast, hue, saturation);
	}
	YUVEFFECTS_Y(Yold[offset].w, u_value, v_value, Y[offset].w, brightness, contrast, hue, saturation);
}

void launchYUV420Effects(const Npp8u* y, const Npp8u* u, const Npp8u* v, Npp8u* Y, Npp8u* U, Npp8u* V, Npp32f brightness, Npp32f contrast, Npp32f hue, Npp32f saturation, int step_y, int step_uv, NppiSize size, cudaStream_t stream)
{
	auto mod = size.width % 8;
	if (mod != 0)
	{
		// we would just process few extra pixels - step is anyway bigger than width and is aligned by 512/256
		size.width += 8 - mod;
	}

	auto width = size.width >> 2;
	step_y = step_y >> 2;
	dim3 block(32, 32);
	dim3 grid((width + block.x - 1) / block.x, (size.height + block.y - 1) / block.y);
	yuv420effects << <grid, block, 0, stream >> > (reinterpret_cast<const uchar4*>(y), reinterpret_cast<const Npp8u*>(u), reinterpret_cast<const Npp8u*>(v), reinterpret_cast<uchar4*>(Y), reinterpret_cast<Npp8u*>(U), reinterpret_cast<Npp8u*>(V), brightness, contrast, hue, saturation, width, size.height, step_y, step_uv);
}