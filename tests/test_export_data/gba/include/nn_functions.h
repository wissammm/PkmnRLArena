#ifndef NN_FUNCTIONS_H
#define NN_FUNCTIONS_H

#define IN_EWRAM __attribute__((section(".ewram")))
#define IN_IWRAM __attribute__((section(".iwram")))

#ifndef CP_FASTSET_COPY
  #define CP_FASTSET_COPY 0x00000000
#endif

#ifndef CPU_FAST_SET_SRC_FIXED
  #define CPU_FAST_SET_SRC_FIXED 0x01000000
#endif

/* mask to keep only the count bits (safe) */
#define CPU_FAST_SET_COUNT_MASK 0x001FFFFF

#define CPU_FAST_SET_COUNT_WORDS(bytes) \
    (((u32)((bytes) / 4)) & CPU_FAST_SET_COUNT_MASK)


#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define CLAMP(x, low, high) (((x) > (high)) ? (high) : (((x) < (low)) ? (low) : (x)))

#define WEIGHT_CHUNK_BYTES 32
#define WEIGHT_BUFFER_BYTES 32
IN_IWRAM static int8_t weight_buffer[WEIGHT_BUFFER_BYTES] __attribute__((aligned(4)));


static inline void relu_int8_t(int8_t* input, int8_t* output, int size) {
    for(int i = 0; i < size; i++) {
        output[i] = input[i] > 0 ? input[i] : 0;
    }
}

static inline void fc_int8_t(int8_t* input, int8_t* output, const int8_t* weights, const int32_t* biases, int in_size, int out_size) {
    for (int i = 0; i < out_size; i++) {
        int32_t sum = biases[i];
        
        for (int j = 0; j < in_size; j++) {
            sum += (int32_t)weights[i * in_size + j] * input[j];
        }

        // Clip
        if (sum > 127) sum = 127;
        if (sum < -128) sum = -128;
        
        output[i] = (int8_t)sum;
    }
}

// static inline void qgemm_int8_t(const int8_t* input, int8_t* output,
//                  const int8_t* weights, const int32_t* biases,
//                  const int input_size, const int output_size,
//                  const int32_t multiplier, const int32_t shift)
// {
//     for (int out = 0; out < output_size; ++out) {
//         int32_t acc = biases ? biases[out] : 0;
//         /* weights layout: weights[out * input_size + k] */
//         const int8_t *wbase = weights + out * input_size;

//         int k = 0;
//         /* chunks complets de 32 bytes */
//         for (; k + WEIGHT_CHUNK_BYTES <= input_size; k += WEIGHT_CHUNK_BYTES) {
//             // copy 32 bytes ROM -> IWRAM
//             CpuFastSet((const void*)(wbase + k),
//                 (void*)weight_buffer,
//                 CP_FASTSET_COPY | CPU_FAST_SET_COUNT_WORDS(WEIGHT_CHUNK_BYTES));

//             const int8_t *wb = weight_buffer;
//             const int8_t *ib = input + k;

//             /* MAC 32 éléments */
//             for (int j = 0; j < WEIGHT_CHUNK_BYTES; ++j) {
//                 acc += (int32_t)wb[j] * (int32_t)ib[j];
//             }
//         }

//         /* remainder < 32 bytes */
//         for (; k < input_size; ++k) {
//             acc += (int32_t)wbase[k] * (int32_t)input[k];
//         }

//         /* Requantize with rounding towards nearest (your method) */
//         int64_t scaled = (int64_t)acc * (int64_t)multiplier;
//         if (shift > 0) {
//             /* rounding: add 1 << (shift-1) */
//             scaled += ((int64_t)1 << (shift - 1));
//             output[out] = (int8_t)CLAMP((int32_t)(scaled >> shift), -128, 127);
//         } else if (shift == 0) {
//             output[out] = (int8_t)CLAMP((int32_t)scaled, -128, 127);
//         } else { /* shift < 0 -> left shift */
//             int s = -shift;
//             /* beware overflow if s large, but assume shifts small */
//             int32_t shifted = (int32_t)(scaled << s);
//             output[out] = (int8_t)CLAMP(shifted, -128, 127);
//         }
//     }
// }

// A simpler version without DMA optimization
static inline void qgemm_int8_t(const int8_t* input, int8_t* output,
                 const int8_t* weights, const int32_t* biases,
                 const int input_size, const int output_size,
                 const int32_t multiplier, const int32_t shift) {
    
    for (int out_idx = 0; out_idx < output_size; out_idx++) {
        int32_t acc = biases[out_idx];
        
        const int8_t* w = weights + out_idx * input_size;
        
        for (int in_idx = 0; in_idx < input_size; in_idx++) {
            int32_t input_val = input[in_idx];
            int32_t weight_val = w[in_idx];
            
            acc += input_val * weight_val;
        }
        
        int64_t scaled = (int64_t)acc * multiplier;
        
        if (shift > 0) {
            scaled += (1LL << (shift - 1));
        }
        
        int32_t requantized = (int32_t)(scaled >> shift);
        
        if (requantized > 127) requantized = 127;
        if (requantized < -128) requantized = -128;
        
        output[out_idx] = (int8_t)requantized;
    }
}


#endif // NN_FUNCTIONS_H