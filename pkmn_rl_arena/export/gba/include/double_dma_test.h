#include <gba_console.h>
#include <gba_video.h>
#include <gba_interrupt.h>
#include <gba_systemcalls.h>
#include <gba_timers.h>
#include <gba_types.h>
#include <gba_dma.h>
#include <stdio.h>
#include "test_array.h"

#define IN_EWRAM __attribute__((section(".ewram")))
#define IN_IWRAM __attribute__((section(".iwram")))

#define INPUT_SIZE 1024
#define OUTPUT_SIZE 256
#define CHECK_VERIFY 16
#define NB_ACCESS 5
#define DMA_CHUNK 512 // max IWram chunk

#ifndef TIMER_DIV_64
#define TIMER_DIV_64 (2 << 0)
#endif
#ifndef TIMER_ENABLE
#define TIMER_ENABLE (1 << 7)
#endif

#ifndef DMA_32BIT
#define DMA_32BIT (1 << 26)
#endif


// ---------------- TIMER ----------------
static inline void timer_start() {
    REG_TM3CNT_L = 0;               // reset counter
    REG_TM3CNT_H = TIMER_ENABLE | TIMER_DIV_64;  // start timer
}

static inline u32 timer_stop() {
    u16 count = REG_TM3CNT_L;       // read 16-bit counter
    REG_TM3CNT_H &= ~TIMER_ENABLE;  // stop timer
    return (u32)count;
}

// ---------------- DMA ----------------
static inline void dma_wait_channel(int dma_id) {
    switch(dma_id) {
        case 0: while(REG_DMA0CNT & DMA_ENABLE); break;
        case 1: while(REG_DMA1CNT & DMA_ENABLE); break;
        case 2: while(REG_DMA2CNT & DMA_ENABLE); break;
        case 3: while(REG_DMA3CNT & DMA_ENABLE); break;
    }
}

static inline void dma_copy_sync(int dma_id, const void* src, void* dst, int size) {
    // Validate alignment for 32-bit transfers
    if (((u32)src & 3) || ((u32)dst & 3) || (size & 3)) {
        return;
    }
    
    dma_wait_channel(dma_id);
    
    u32 count = size / 4;
    u32 control = count | DMA_ENABLE | DMA_32BIT | DMA_SRC_INC | DMA_DST_INC;
    
    switch(dma_id) {
        case 0:
            REG_DMA0SAD = (u32)src;
            REG_DMA0DAD = (u32)dst;
            REG_DMA0CNT = control;
            while(REG_DMA0CNT & DMA_ENABLE);
            break;
        case 1:
            REG_DMA1SAD = (u32)src;
            REG_DMA1DAD = (u32)dst;
            REG_DMA1CNT = control;
            while(REG_DMA1CNT & DMA_ENABLE);
            break;
        case 2:
            REG_DMA2SAD = (u32)src;
            REG_DMA2DAD = (u32)dst;
            REG_DMA2CNT = control;
            while(REG_DMA2CNT & DMA_ENABLE);
            break;
        case 3:
            REG_DMA3SAD = (u32)src;
            REG_DMA3DAD = (u32)dst;
            REG_DMA3CNT = control;
            while(REG_DMA3CNT & DMA_ENABLE);
            break;
    }
}
// ---------------- INPUT / OUTPUT ----------------
static int8_t input_data[INPUT_SIZE] IN_EWRAM;
static int8_t output_data[OUTPUT_SIZE] IN_EWRAM;
static int8_t bufferIW1[512] IN_IWRAM;
static int8_t bufferEW1[512] IN_EWRAM;
static int8_t output_data[OUTPUT_SIZE] IN_EWRAM;

void init_qgemm_data(){
    for(int i=0;i<INPUT_SIZE;i++) input_data[i]=(int8_t)(i & 0x7F);
    for(int i=0;i<OUTPUT_SIZE;i++) output_data[i]=0;
}

// ===============================================================
// Baseline qGEMM (no DMA)
// ===============================================================
static inline void qgemm_int8_t(const int8_t* input, int8_t* output,
                 const int8_t* weights, const int32_t* biases,
                 const int input_size, const int output_size,
                 const int32_t multiplier, const int32_t shift) {

    for(int out_idx=0; out_idx<output_size; out_idx++){
        int32_t acc = biases[out_idx];
        const int8_t* w = weights + out_idx*input_size;
        for(int in_idx=0; in_idx<input_size; in_idx++)
            acc += (int32_t)input[in_idx] * (int32_t)w[in_idx];

        int64_t scaled = (int64_t)acc*multiplier;
        if(shift>0) scaled += (1LL<<(shift-1));
        int32_t requantized = (int32_t)(scaled>>shift);
        if(requantized>127) requantized=127;
        if(requantized<-128) requantized=-128;
        output[out_idx] = (int8_t)requantized;
    }
}

// ===============================================================
// Double-buffered qGEMM with DMA prefetching
// ===============================================================
static inline void qgemm_int8_dma(const int8_t* input, int8_t* output,
                 const int8_t* weights, const int32_t* biases,
                 const int input_size, const int output_size,
                 const int32_t multiplier, const int32_t shift) {

    const int CHUNK_SIZE = 512; // IWRAM buffer size (weights chunk)
    
    // Process each output neuron
    for(int out_idx = 0; out_idx < output_size; out_idx++) {
        int32_t acc = biases[out_idx];
        const int8_t* w = weights + out_idx * input_size;
        
        int chunks_needed = (input_size + CHUNK_SIZE - 1) / CHUNK_SIZE;
        
        // Pre-load first chunk from ROM to EWRAM via DMA
        int first_chunk_size = (input_size < CHUNK_SIZE) ? input_size : CHUNK_SIZE;
        dma_copy_sync(3, w, bufferEW1, first_chunk_size);
        
        for(int chunk_idx = 0; chunk_idx < chunks_needed; chunk_idx++) {
            int offset = chunk_idx * CHUNK_SIZE;
            int remaining = input_size - offset;
            int current_chunk_size = (remaining < CHUNK_SIZE) ? remaining : CHUNK_SIZE;
            
            // Move current chunk from EWRAM to IWRAM (fast internal copy)
            for(int i = 0; i < current_chunk_size; i++) {
                bufferIW1[i] = bufferEW1[i];
            }
            
            // Start DMA for NEXT chunk (ROM → EWRAM) while we process current
            if(chunk_idx + 1 < chunks_needed) {
                int next_offset = (chunk_idx + 1) * CHUNK_SIZE;
                int next_remaining = input_size - next_offset;
                int next_chunk_size = (next_remaining < CHUNK_SIZE) ? next_remaining : CHUNK_SIZE;
                
                // Async DMA transfer (non-blocking)
                u32 count = (next_chunk_size + 3) / 4; // Round up for 32-bit
                REG_DMA3SAD = (u32)(w + next_offset);
                REG_DMA3DAD = (u32)bufferEW1;
                REG_DMA3CNT = count | DMA_ENABLE | DMA_32BIT | DMA_SRC_INC | DMA_DST_INC;
            }
            
            // Process current chunk in IWRAM (CPU works in parallel with DMA)
            for(int i = 0; i < current_chunk_size; i++) {
                acc += (int32_t)input[offset + i] * (int32_t)bufferIW1[i];
            }
            
            // Wait for DMA to finish before next iteration
            if(chunk_idx + 1 < chunks_needed) {
                while(REG_DMA3CNT & DMA_ENABLE);
            }
        }
        
        // Quantization
        int64_t scaled = (int64_t)acc * multiplier;
        if(shift > 0) scaled += (1LL << (shift - 1));
        int32_t requantized = (int32_t)(scaled >> shift);
        if(requantized > 127) requantized = 127;
        if(requantized < -128) requantized = -128;
        output[out_idx] = (int8_t)requantized;
    }
}

// ===============================================================
// Alternative: Input prefetching (if input changes per iteration)
// ===============================================================
static inline void qgemm_int8_dma_input(const int8_t* input, int8_t* output,
                 const int8_t* weights, const int32_t* biases,
                 const int input_size, const int output_size,
                 const int32_t multiplier, const int32_t shift) {

    const int CHUNK_SIZE = 512;
    
    // Prefetch input to IWRAM once (if input is in ROM/EWRAM)
    int input_chunks = (input_size + CHUNK_SIZE - 1) / CHUNK_SIZE;
    
    for(int out_idx = 0; out_idx < output_size; out_idx++) {
        int32_t acc = biases[out_idx];
        const int8_t* w = weights + out_idx * input_size;
        
        // Process in chunks with double buffering
        for(int chunk_idx = 0; chunk_idx < input_chunks; chunk_idx++) {
            int offset = chunk_idx * CHUNK_SIZE;
            int remaining = input_size - offset;
            int chunk_size = (remaining < CHUNK_SIZE) ? remaining : CHUNK_SIZE;
            
            // Load input chunk to IWRAM
            for(int i = 0; i < chunk_size; i++) {
                bufferIW1[i] = input[offset + i];
            }
            
            // Start DMA for weights chunk (ROM → EWRAM)
            u32 count = (chunk_size + 3) / 4;
            REG_DMA3SAD = (u32)(w + offset);
            REG_DMA3DAD = (u32)bufferEW1;
            REG_DMA3CNT = count | DMA_ENABLE | DMA_32BIT | DMA_SRC_INC | DMA_DST_INC;
            
            // Wait for weights to arrive
            while(REG_DMA3CNT & DMA_ENABLE);
            
            // Compute dot product
            for(int i = 0; i < chunk_size; i++) {
                acc += (int32_t)bufferIW1[i] * (int32_t)bufferEW1[i];
            }
        }
        
        // Quantization
        int64_t scaled = (int64_t)acc * multiplier;
        if(shift > 0) scaled += (1LL << (shift - 1));
        int32_t requantized = (int32_t)(scaled >> shift);
        if(requantized > 127) requantized = 127;
        if(requantized < -128) requantized = -128;
        output[out_idx] = (int8_t)requantized;
    }
}

// Add these buffers to your global declarations

static int8_t bufferEW2[512] IN_EWRAM;    
static int8_t inputIW[1024] IN_IWRAM;     

// ===============================================================
// FASTEST: Triple-buffered with input in IWRAM
// ===============================================================
static inline void qgemm_int8_dma_v3(const int8_t* input, int8_t* output,
                 const int8_t* weights, const int32_t* biases,
                 const int input_size, const int output_size,
                 const int32_t multiplier, const int32_t shift) {

    const int CHUNK_SIZE = 512;
    
    // Copy input to IWRAM once (avoids EWRAM bus conflicts)
    // Use DMA for fast copy
    int input_size_aligned = (input_size + 3) & ~3;
    if(input_size <= 1024) {  // If input fits in IWRAM
        dma_copy_sync(0, input, bufferIW1, input_size_aligned);
    }
    const int8_t* input_ptr = (input_size <= 1024) ? bufferIW1 : input;
    
    // Process each output neuron
    for(int out_idx = 0; out_idx < output_size; out_idx++) {
        int32_t acc = biases[out_idx];
        const int8_t* w = weights + out_idx * input_size;
        
        int chunks_needed = (input_size + CHUNK_SIZE - 1) / CHUNK_SIZE;
        
        // Pre-load first chunk from ROM to EWRAM via DMA
        int first_chunk_size = (input_size < CHUNK_SIZE) ? input_size : CHUNK_SIZE;
        int first_size_aligned = (first_chunk_size + 3) & ~3;
        dma_copy_sync(3, w, bufferEW1, first_size_aligned);
        
        for(int chunk_idx = 0; chunk_idx < chunks_needed; chunk_idx++) {
            int offset = chunk_idx * CHUNK_SIZE;
            int remaining = input_size - offset;
            int current_chunk_size = (remaining < CHUNK_SIZE) ? remaining : CHUNK_SIZE;
            
            // Determine which EWRAM buffer was just filled
            int8_t* current_ew = (chunk_idx & 1) ? bufferEW2 : bufferEW1;
            int8_t* next_ew    = (chunk_idx & 1) ? bufferEW1 : bufferEW2;

            // Start DMA for NEXT chunk (ROM → EWRAM) FIRST
            if(chunk_idx + 1 < chunks_needed) {
                int next_offset = (chunk_idx + 1) * CHUNK_SIZE;
                int next_remaining = input_size - next_offset;
                int next_chunk_size = (next_remaining < CHUNK_SIZE) ? next_remaining : CHUNK_SIZE;
                int next_size_aligned = (next_chunk_size + 3) & ~3;
                
                // Start async DMA to alternate buffer
                REG_DMA3SAD = (u32)(w + next_offset);
                REG_DMA3DAD = (u32)next_ew;
                REG_DMA3CNT = (next_size_aligned / 4) | DMA_ENABLE | DMA_32BIT | DMA_SRC_INC | DMA_DST_INC;
            }
            
            // Process current chunk
            // Unrolled loop for 4x speedup (process 4 elements at a time)
            int i = 0;
            int limit = current_chunk_size & ~3; // Round down to multiple of 4
            
            for(; i < limit; i += 4) {
                int32_t w0 = current_ew[i];
                int32_t w1 = current_ew[i+1];
                int32_t w2 = current_ew[i+2];
                int32_t w3 = current_ew[i+3];
                
                int32_t in0 = input_ptr[offset + i];
                int32_t in1 = input_ptr[offset + i + 1];
                int32_t in2 = input_ptr[offset + i + 2];
                int32_t in3 = input_ptr[offset + i + 3];
                
                acc += w0 * in0 + w1 * in1 + w2 * in2 + w3 * in3;
            }
            
            // Handle remaining elements
            for(; i < current_chunk_size; i++) {
                acc += (int32_t)input_ptr[offset + i] * (int32_t)current_ew[i];
            }
            
            // Wait for DMA to finish before next iteration
            if(chunk_idx + 1 < chunks_needed) {
                while(REG_DMA3CNT & DMA_ENABLE);
            }
        }
        
        // Quantization
        int64_t scaled = (int64_t)acc * multiplier;
        if(shift > 0) scaled += (1LL << (shift - 1));
        int32_t requantized = (int32_t)(scaled >> shift);
        if(requantized > 127) requantized = 127;
        if(requantized < -128) requantized = -128;
        output[out_idx] = (int8_t)requantized;
    }
}


static inline int dma3_start_async_aligned(const int8_t* src, int8_t* dst, int length,
                                           const int8_t** tail_src, int8_t** tail_dst, int* tail_len) {
    int copied = 0;
    while (copied < length && ((u32)(src + copied) & 3)) {
        dst[copied] = src[copied];
        ++copied;
    }
    int aligned = (length - copied) & ~3;
    if (aligned) {
        REG_DMA3SAD = (u32)(src + copied);
        REG_DMA3DAD = (u32)(dst + copied);
        REG_DMA3CNT = (aligned / 4) | DMA_ENABLE | DMA_32BIT | DMA_SRC_INC | DMA_DST_INC;
        copied += aligned;
    }
    *tail_len = length - copied;
    *tail_src = src + copied;
    *tail_dst = dst + copied;
    return aligned != 0;
}

static inline void qgemm_int8_dma_v3_all_inputs(const int8_t* input, int8_t* output,
                 const int8_t* weights, const int32_t* biases,
                 const int input_size, const int output_size,
                 const int32_t multiplier, const int32_t shift) {

    const int CHUNK_SIZE = 512;

    int input_prefetched = 0;
    const int8_t* cached_input = input;
    if (input_size <= (int)sizeof(inputIW)) {
        int bulk = input_size & ~3;
        if (bulk) {
            dma_copy_sync(0, input, inputIW, bulk);
        }
        for (int i = bulk; i < input_size; ++i) {
            inputIW[i] = input[i];
        }
        cached_input = inputIW;
        input_prefetched = 1;
    }

    for (int out_idx = 0; out_idx < output_size; ++out_idx) {
        int32_t acc = biases[out_idx];
        const int8_t* w = weights + out_idx * input_size;

        int chunks_needed = (input_size + CHUNK_SIZE - 1) / CHUNK_SIZE;

        int first_chunk_size = (input_size < CHUNK_SIZE) ? input_size : CHUNK_SIZE;
        int first_dma_bytes = first_chunk_size & ~3;
        if (first_dma_bytes) {
            dma_copy_sync(3, w, bufferEW1, first_dma_bytes);
        }
        for (int i = first_dma_bytes; i < first_chunk_size; ++i) {
            bufferEW1[i] = w[i];
        }

        for (int chunk_idx = 0; chunk_idx < chunks_needed; ++chunk_idx) {
            int offset = chunk_idx * CHUNK_SIZE;
            int remaining = input_size - offset;
            int current_chunk_size = (remaining < CHUNK_SIZE) ? remaining : CHUNK_SIZE;

            int8_t* current_ew = (chunk_idx & 1) ? bufferEW2 : bufferEW1;
            int8_t* next_ew    = (chunk_idx & 1) ? bufferEW1 : bufferEW2;

            const int8_t* chunk_input;
            if (input_prefetched) {
                chunk_input = cached_input + offset;
            } else {
                int input_dma_bytes = current_chunk_size & ~3;
                if (input_dma_bytes) {
                    dma_copy_sync(0, input + offset, bufferIW1, input_dma_bytes);
                }
                for (int i = input_dma_bytes; i < current_chunk_size; ++i) {
                    bufferIW1[i] = input[offset + i];
                }
                chunk_input = bufferIW1;
            }

            int i = 0;
            int limit = current_chunk_size & ~3;
            for (; i < limit; i += 4) {
                acc += (int32_t)current_ew[i]     * (int32_t)chunk_input[i]
                     + (int32_t)current_ew[i + 1] * (int32_t)chunk_input[i + 1]
                     + (int32_t)current_ew[i + 2] * (int32_t)chunk_input[i + 2]
                     + (int32_t)current_ew[i + 3] * (int32_t)chunk_input[i + 3];
            }
            for (; i < current_chunk_size; ++i) {
                acc += (int32_t)current_ew[i] * (int32_t)chunk_input[i];
            }

            int dma_inflight = 0;
            int pending_tail = 0;
            const int8_t* pending_tail_src = NULL;
            int8_t* pending_tail_dst = NULL;

            if (chunk_idx + 1 < chunks_needed) {
                int next_offset = (chunk_idx + 1) * CHUNK_SIZE;
                int next_remaining = input_size - next_offset;
                int next_chunk_size = (next_remaining < CHUNK_SIZE) ? next_remaining : CHUNK_SIZE;

                int next_dma_bytes = next_chunk_size & ~3;
                int next_tail = next_chunk_size - next_dma_bytes;

                if (next_dma_bytes) {
                    dma_inflight = 1;
                    pending_tail = next_tail;
                    pending_tail_src = w + next_offset + next_dma_bytes;
                    pending_tail_dst = next_ew + next_dma_bytes;

                    REG_DMA3SAD = (u32)(w + next_offset);
                    REG_DMA3DAD = (u32)next_ew;
                    REG_DMA3CNT = (next_dma_bytes / 4) | DMA_ENABLE | DMA_32BIT | DMA_SRC_INC | DMA_DST_INC;
                } else {
                    for (int t = 0; t < next_tail; ++t) {
                        next_ew[t] = w[next_offset + t];
                    }
                }
            }

            if (dma_inflight) {
                while (REG_DMA3CNT & DMA_ENABLE);
                if (pending_tail) {
                    for (int t = 0; t < pending_tail; ++t) {
                        pending_tail_dst[t] = pending_tail_src[t];
                    }
                }
            }
        }

        int64_t scaled = (int64_t)acc * multiplier;
        if (shift > 0) scaled += (1LL << (shift - 1));
        int32_t requantized = (int32_t)(scaled >> shift);
        if (requantized > 127) requantized = 127;
        if (requantized < -128) requantized = -128;
        output[out_idx] = (int8_t)requantized;
    }
}

// ===============================================================
// ALTERNATIVE: If input doesn't fit in IWRAM, use streaming
// ===============================================================
static inline void qgemm_int8_dma_streaming(const int8_t* input, int8_t* output,
                 const int8_t* weights, const int32_t* biases,
                 const int input_size, const int output_size,
                 const int32_t multiplier, const int32_t shift) {

    const int CHUNK_SIZE = 256; // Smaller chunks for both input and weights
    
    for(int out_idx = 0; out_idx < output_size; out_idx++) {
        int32_t acc = biases[out_idx];
        const int8_t* w = weights + out_idx * input_size;
        
        int chunks_needed = (input_size + CHUNK_SIZE - 1) / CHUNK_SIZE;
        
        for(int chunk_idx = 0; chunk_idx < chunks_needed; chunk_idx++) {
            int offset = chunk_idx * CHUNK_SIZE;
            int remaining = input_size - offset;
            int chunk_size = (remaining < CHUNK_SIZE) ? remaining : CHUNK_SIZE;
            int size_aligned = (chunk_size + 3) & ~3;
            
            // Load input chunk to IWRAM first (DMA0)
            REG_DMA0SAD = (u32)(input + offset);
            REG_DMA0DAD = (u32)bufferIW1;
            REG_DMA0CNT = (size_aligned / 4) | DMA_ENABLE | DMA_32BIT | DMA_SRC_INC | DMA_DST_INC;
            
            // Load weights chunk to EWRAM (DMA3) - can run in parallel
            REG_DMA3SAD = (u32)(w + offset);
            REG_DMA3DAD = (u32)bufferEW1;
            REG_DMA3CNT = (size_aligned / 4) | DMA_ENABLE | DMA_32BIT | DMA_SRC_INC | DMA_DST_INC;
            
            // Wait for both DMAs
            while(REG_DMA0CNT & DMA_ENABLE);
            while(REG_DMA3CNT & DMA_ENABLE);
            
            // Compute with unrolling
            int i = 0;
            int limit = chunk_size & ~3;
            for(; i < limit; i += 4) {
                acc += (int32_t)bufferIW1[i] * (int32_t)bufferEW1[i]
                     + (int32_t)bufferIW1[i+1] * (int32_t)bufferEW1[i+1]
                     + (int32_t)bufferIW1[i+2] * (int32_t)bufferEW1[i+2]
                     + (int32_t)bufferIW1[i+3] * (int32_t)bufferEW1[i+3];
            }
            for(; i < chunk_size; i++) {
                acc += (int32_t)bufferIW1[i] * (int32_t)bufferEW1[i];
            }
        }
        
        // Quantization
        int64_t scaled = (int64_t)acc * multiplier;
        if(shift > 0) scaled += (1LL << (shift - 1));
        int32_t requantized = (int32_t)(scaled >> shift);
        if(requantized > 127) requantized = 127;
        if(requantized < -128) requantized = -128;
        output[out_idx] = (int8_t)requantized;
    }
}

// ===============================================================
// Benchmark all versions
// ===============================================================
void run_qgemm_benchmarks_extended(){
    iprintf("\n=== qGEMM Benchmarks ===\n");

    init_qgemm_data();
    const int32_t* biases = (const int32_t*)&tests_array[0];
    const int8_t* weights = (const int8_t*)&tests_array[OUTPUT_SIZE];
    const int32_t multiplier = 1073741824;
    const int32_t shift = 10;

    // --- Baseline ---
    timer_start();
    for(int i=0; i<NB_ACCESS; i++)
        qgemm_int8_t(input_data, output_data, weights, biases,
                     INPUT_SIZE, OUTPUT_SIZE, multiplier, shift);
    u32 cycles1 = timer_stop();
    int32_t checksum1 = 0;
    for(int i=0; i<CHECK_VERIFY; i++) checksum1 += output_data[i];
    iprintf("Baseline:    %lu cycles, sum=%d\n", cycles1, checksum1);

    // --- V3: Triple-buffered ---
    for(int i=0; i<OUTPUT_SIZE; i++) output_data[i] = 0;
    timer_start();
    for(int i=0; i<NB_ACCESS; i++)
        qgemm_int8_dma_v3(input_data, output_data, weights, biases,
                          INPUT_SIZE, OUTPUT_SIZE, multiplier, shift);
    u32 cycles2 = timer_stop();
    int32_t checksum2 = 0;
    for(int i=0; i<CHECK_VERIFY; i++) checksum2 += output_data[i];
    iprintf("DMA Triple:  %lu cycles, sum=%d\n", cycles2, checksum2);
    
    // Calculate speedup as percentage (integer math)
    u32 speedup_x100 = (cycles1 * 100) / cycles2;
    iprintf("Speedup: %lu.%02lux\n", speedup_x100/100, speedup_x100%100);
    
    // --- Streaming version ---
    for(int i=0; i<OUTPUT_SIZE; i++) output_data[i] = 0;
    timer_start();
    for(int i=0; i<NB_ACCESS; i++)
        qgemm_int8_dma_streaming(input_data, output_data, weights, biases,
                                 INPUT_SIZE, OUTPUT_SIZE, multiplier, shift);
    u32 cycles3 = timer_stop();
    int32_t checksum3 = 0;
    for(int i=0; i<CHECK_VERIFY; i++) checksum3 += output_data[i];
    iprintf("DMA Stream:  %lu cycles, sum=%d\n", cycles3, checksum3);
    speedup_x100 = (cycles1 * 100) / cycles3;
    iprintf("Speedup: %lu.%02lux\n", speedup_x100/100, speedup_x100%100);
    // --- All Inputs ---
    for(int i=0; i<OUTPUT_SIZE; i++) output_data[i] = 0;
    timer_start();
    for(int i=0; i<NB_ACCESS; i++)
        qgemm_int8_dma_v3_all_inputs(input_data, output_data, weights, biases,
                                 INPUT_SIZE, OUTPUT_SIZE, multiplier, shift);
    u32 cycles4 = timer_stop();
    int32_t checksum4 = 0;
    for(int i=0; i<CHECK_VERIFY; i++) checksum4 += output_data[i];
    iprintf("All Inputs:  %lu cycles, sum=%d\n", cycles4, checksum4);
    speedup_x100 = (cycles1 * 100) / cycles4;
    iprintf("Speedup: %lu.%02lux\n", speedup_x100/100, speedup_x100%100);

    

}