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
#define INPUT_SIZE 256
#define OUTPUT_SIZE 64
#define NB_ACCESS 10

#ifndef TIMER_DIV_64
#define TIMER_DIV_64 (2 << 0)
#endif
#ifndef TIMER_ENABLE
#define TIMER_ENABLE (1 << 7)
#endif

static inline void timer_start() {
    REG_TM3CNT_L = 0;
    REG_TM3CNT_H = TIMER_ENABLE | TIMER_DIV_64;
}
static inline u32 timer_stop() {
    REG_TM3CNT_H &= ~TIMER_ENABLE;
    return REG_TM3CNT_L;
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

/* ===============================================================
   qgemm_int8_t — baseline version
   =============================================================== */
static inline void qgemm_int8_t(const int8_t* input, int8_t* output,
                 const int8_t* weights, const int32_t* biases,
                 const int input_size, const int output_size,
                 const int32_t multiplier, const int32_t shift)
{
    for (int out_idx = 0; out_idx < output_size; out_idx++) {
        int32_t acc = biases[out_idx];
        const int8_t* w = weights + out_idx * input_size;

        for (int in_idx = 0; in_idx < input_size; in_idx++) {
            acc += (int32_t)input[in_idx] * (int32_t)w[in_idx];
        }

        int64_t scaled = (int64_t)acc * multiplier;
        if (shift > 0) scaled += (1LL << (shift - 1));
        int32_t requantized = (int32_t)(scaled >> shift);
        if (requantized > 127) requantized = 127;
        if (requantized < -128) requantized = -128;
        output[out_idx] = (int8_t)requantized;
    }
}

/* ===============================================================
   Test setup
   =============================================================== */

static int8_t input_data[INPUT_SIZE] IN_EWRAM;
static int8_t output_data[OUTPUT_SIZE] IN_EWRAM;

void init_qgemm_data() {
    // Fill input with deterministic values 0..255
    for (int i = 0; i < INPUT_SIZE; i++)
        input_data[i] = (int8_t)(i & 0x7F);

    // Clear output
    for (int i = 0; i < OUTPUT_SIZE; i++)
        output_data[i] = 0;
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

// ===============================================================
// Add to benchmark runner
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
    u32 checksum1 = 0;
    for(int i=0; i<OUTPUT_SIZE; i++) checksum1 += output_data[i];
    iprintf("Baseline:    Cycles=%lu, Checksum=%lu\n", cycles1, checksum1);

    // --- DMA Optimized ---
    for(int i=0; i<OUTPUT_SIZE; i++) output_data[i] = 0;
    timer_start();
    for(int i=0; i<NB_ACCESS; i++)
        qgemm_int8_dma(input_data, output_data, weights, biases,
                       INPUT_SIZE, OUTPUT_SIZE, multiplier, shift);
    u32 cycles2 = timer_stop();
    u32 checksum2 = 0;
    for(int i=0; i<OUTPUT_SIZE; i++) checksum2 += output_data[i];
    iprintf("DMA Double:  Cycles=%lu, Checksum=%lu\n", cycles2, checksum2);
    
    float speedup = (float)cycles1 / cycles2;
    iprintf("Speedup: %.2fx\n", speedup);
}
