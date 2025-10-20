#ifndef TESTS_H
#define TESTS_H

#include <gba_console.h>
#include <gba_video.h>
#include <gba_interrupt.h>
#include <gba_systemcalls.h>
#include <gba_input.h>
#include <gba_types.h>
#include <gba_timers.h>
#include <gba_dma.h>
#include <gba.h>
#include <stdio.h>
#include "test_array.h"

// Timer definitions
#ifndef TIMER_DIV_64
#define TIMER_DIV_64 (2 << 0)
#endif
#ifndef TIMER_ENABLE
#define TIMER_ENABLE (1 << 7)
#endif
#ifndef TIMER_IRQ
#define TIMER_IRQ (1 << 6)
#endif

// CPU Fast Set definitions
#ifndef CPU_FAST_SET_32BIT
#define CPU_FAST_SET_32BIT (1 << 24)
#endif

// DMA definitions
#ifndef DMA_32BIT
#define DMA_32BIT (1 << 26)
#endif

#define IN_EWRAM __attribute__((section(".ewram")))
#define IN_IWRAM __attribute__((section(".iwram")))

#define NB_ACCESS 5  // Number of times to repeat each test for averaging

// Global accumulator to prevent optimization
volatile u32 global_result = 0;

static inline u32 timer_start(int timer_id)
{
    REG_TM3CNT_L = 0;
    REG_TM3CNT_H = 0;
    REG_TM3CNT_H = TIMER_ENABLE | TIMER_DIV_64;
    return 0;
}

static inline u32 timer_stop(int timer_id)
{
    REG_TM3CNT_H = REG_TM3CNT_H & ~TIMER_ENABLE;
    return REG_TM3CNT_L;
}

static void __attribute__((section(".iwram_text"), noinline))
process_data_chunk(const int8_t* data, int size, int global_offset, volatile u32* result)
{
    volatile u32 sum = 0;
    volatile u32 product = 1;
    volatile u32 hash = 0;

    for (int i = 0; i < size; i++) {
        int idx = global_offset + i;  

        sum += (u32)data[i] * (idx + 1);
        hash = ((hash << 3) + hash) + (u32)data[i] + idx;

        if (sum > 0xFFFF) sum = sum % 0x8000 + 1000;
        if (hash > 0xFFFF) hash = hash % 0x7FFF + 500;

        if ((idx & 3) == 0) {
            product = (product * ((u32)data[i] + 1)) & 0xFFFF;
            if (product == 0) product = 1;
        }

        if (data[i] > 0) {
            sum += idx * 2;
        } else {
            sum -= idx;
        }
    }

    *result += (sum ^ hash ^ product) & 0xFFFF;
}

// Test 1: Direct ROM access (baseline)
static inline u32 test_load_from_rom()
{
    volatile u32 total_result = 0;

    for (int i = 0; i < NB_ACCESS; ++i)
    {
        for (int j = 0; j < 32768; j += 32)
        {
            process_data_chunk(&tests_array[j], 32, j, &total_result);
        }
    }

    global_result += total_result;
    return total_result & 0xFF;
}

int8_t buffer[32] IN_IWRAM __attribute__((aligned(4)));

// Test 2: CpuFastSet ROM -> IWRAM
static inline u32 test_load_from_rom_with_cpuFastSet()
{
    volatile u32 total_result = 0;

    for (int i = 0; i < NB_ACCESS; ++i)
    {
        for (int j = 0; j < 32768; j += 32)
        {
            CpuFastSet(&tests_array[j], buffer, (32 / 4) | CPU_FAST_SET_32BIT);
            process_data_chunk(buffer, 32, j, &total_result);  // ✅ j = global offset
        }
    }

    global_result += total_result;
    return total_result & 0xFF;
}

// Test 3: Single DMA ROM -> IWRAM
static inline u32 test_load_from_rom_to_iwram_with_dma()
{
    volatile u32 total_result = 0;

    for (int i = 0; i < NB_ACCESS; ++i)
    {
        for (int j = 0; j < 32768; j += 32)
        {
            REG_DMA3SAD = (u32)&tests_array[j];
            REG_DMA3DAD = (u32)buffer;
            REG_DMA3CNT = (32 / 4) | DMA_ENABLE | DMA_32BIT | DMA_SRC_INC | DMA_DST_INC;
            while (REG_DMA3CNT & DMA_ENABLE);

            process_data_chunk(buffer, 32, j, &total_result);  // ✅ j = global offset
        }
    }

    global_result += total_result;
    return total_result & 0xFF;
}

int8_t iwram_buf[32] IN_IWRAM __attribute__((aligned(4)));
int8_t ewram_buf[32] IN_EWRAM __attribute__((aligned(4)));
// Test 4: Double-buffered DMA via EWRAM (flawed, but consistent)
static inline u32 test_load_rom_to_ewram_and_ewram_to_iwram_with_dma()
{
    volatile u32 total_result = 0;

    for (int i = 0; i < NB_ACCESS; ++i)
    {
        REG_DMA3SAD = (u32)&tests_array[0];
        REG_DMA3DAD = (u32)iwram_buf;
        REG_DMA3CNT = (32 / 4) | DMA_ENABLE | DMA_32BIT | DMA_SRC_INC | DMA_DST_INC;
        while (REG_DMA3CNT & DMA_ENABLE);

        for (int j = 32; j < 32768; j += 32)
        {
            REG_DMA3SAD = (u32)&tests_array[j];
            REG_DMA3DAD = (u32)ewram_buf;
            REG_DMA3CNT = (32 / 4) | DMA_ENABLE | DMA_32BIT | DMA_SRC_INC | DMA_DST_INC;

            process_data_chunk(iwram_buf, 32, j - 32, &total_result);  // ✅ j-32

            while (REG_DMA3CNT & DMA_ENABLE);

            REG_DMA3SAD = (u32)ewram_buf;
            REG_DMA3DAD = (u32)iwram_buf;
            REG_DMA3CNT = (32 / 4) | DMA_ENABLE | DMA_32BIT | DMA_SRC_INC | DMA_DST_INC;
            while (REG_DMA3CNT & DMA_ENABLE);
        }

        process_data_chunk(iwram_buf, 32, 32768 - 32, &total_result);  // ✅ last chunk
    }

    global_result += total_result;
    return total_result & 0xFF;
}

// Test 5: Dual-channel DMA
static inline u32 test_double_buffered_dma()
{

    volatile u32 total_result = 0;

    for (int i = 0; i < NB_ACCESS; ++i)
    {
        REG_DMA3SAD = (u32)&tests_array[0];
        REG_DMA3DAD = (u32)iwram_buf;
        REG_DMA3CNT = (32 / 4) | DMA_ENABLE | DMA_32BIT | DMA_SRC_INC | DMA_DST_INC;
        while (REG_DMA3CNT & DMA_ENABLE);

        for (int j = 32; j < 32768; j += 32)
        {
            REG_DMA1SAD = (u32)&tests_array[j];
            REG_DMA1DAD = (u32)ewram_buf;
            REG_DMA1CNT = (32 / 4) | DMA_ENABLE | DMA_32BIT | DMA_SRC_INC | DMA_DST_INC;

            process_data_chunk(iwram_buf, 32, j - 32, &total_result);  // ✅

            while (REG_DMA1CNT & DMA_ENABLE);

            REG_DMA3SAD = (u32)ewram_buf;
            REG_DMA3DAD = (u32)iwram_buf;
            REG_DMA3CNT = (32 / 4) | DMA_ENABLE | DMA_32BIT | DMA_SRC_INC | DMA_DST_INC;
            while (REG_DMA3CNT & DMA_ENABLE);
        }

        process_data_chunk(iwram_buf, 32, 32768 - 32, &total_result);
    }

    global_result += total_result;
    return total_result & 0xFF;
}

// Test 6: Realistic double buffering (512-byte chunks, 32-byte subchunks)
int8_t iwram_buf_chunk[512] IN_IWRAM __attribute__((aligned(4)));
int8_t ewram_buf_chunk[512] IN_EWRAM __attribute__((aligned(4)));
static u32 __attribute__((section(".iwram_text"), noinline))
test_realistic_double_buffering()
{
    const int CHUNK_SIZE = 512;
    const int SUB_SIZE = 32;

    volatile u32 total_result = 0;

    for (int pass = 0; pass < NB_ACCESS; ++pass)
    {
        REG_DMA3SAD = (u32)&tests_array[0];
        REG_DMA3DAD = (u32)iwram_buf_chunk;
        REG_DMA3CNT = (CHUNK_SIZE / 4) | DMA_ENABLE | DMA_32BIT | DMA_SRC_INC | DMA_DST_INC;
        while (REG_DMA3CNT & DMA_ENABLE);

        for (int off = CHUNK_SIZE; off < 32768; off += CHUNK_SIZE)
        {
            REG_DMA1SAD = (u32)&tests_array[off];
            REG_DMA1DAD = (u32)ewram_buf_chunk;
            REG_DMA1CNT = (CHUNK_SIZE / 4) | DMA_ENABLE | DMA_32BIT | DMA_SRC_INC | DMA_DST_INC;

            // Process in 32-byte subchunks with correct global offset
            for (int s = 0; s < CHUNK_SIZE; s += SUB_SIZE)
            {
                process_data_chunk(iwram_buf_chunk + s, SUB_SIZE, off - CHUNK_SIZE + s, &total_result);
            }

            while (REG_DMA1CNT & DMA_ENABLE);

            REG_DMA3SAD = (u32)ewram_buf_chunk;
            REG_DMA3DAD = (u32)iwram_buf_chunk;
            REG_DMA3CNT = (CHUNK_SIZE / 4) | DMA_ENABLE | DMA_32BIT | DMA_SRC_INC | DMA_DST_INC;
            while (REG_DMA3CNT & DMA_ENABLE);
        }

        // Final chunk
        for (int s = 0; s < CHUNK_SIZE; s += SUB_SIZE)
        {
            process_data_chunk(iwram_buf_chunk + s, SUB_SIZE, 32768 - CHUNK_SIZE + s, &total_result);
        }
    }

    global_result += total_result;
    return total_result & 0xFF;
}

int8_t buf_a[128] IN_IWRAM __attribute__((aligned(4)));
int8_t buf_b[128] IN_IWRAM __attribute__((aligned(4)));
// Test 7: CPU streaming
static inline u32 test_cpu_streaming()
{
    const int STREAM_SIZE = 128;

    volatile u32 total_result = 0;

    for (int i = 0; i < NB_ACCESS; ++i)
    {
        CpuFastSet(&tests_array[0], buf_a, (STREAM_SIZE/4) | CPU_FAST_SET_32BIT);
        int8_t* current = buf_a;
        int8_t* next = buf_b;

        for (int j = STREAM_SIZE; j < 32768; j += STREAM_SIZE)
        {
            CpuFastSet(&tests_array[j], next, (STREAM_SIZE/4) | CPU_FAST_SET_32BIT);
            process_data_chunk(current, STREAM_SIZE, j - STREAM_SIZE, &total_result);

            volatile u32 stream_hash = 0;
            for (int k = 0; k < STREAM_SIZE; k += 2) {
                stream_hash += current[k] ^ current[k+1];
                stream_hash = (stream_hash << 1) | (stream_hash >> 31);
            }
            total_result += stream_hash & 0xFF;

            int8_t* tmp = current;
            current = next;
            next = tmp;
        }

        process_data_chunk(current, STREAM_SIZE, 32768 - STREAM_SIZE, &total_result);
    }

    global_result += total_result;
    return total_result & 0xFF;
}

// Test 8: Double-buffered DMA3 only
static inline u32 test_double_buffered_dma3_only()
{

    volatile u32 total_result = 0;

    for (int i = 0; i < NB_ACCESS; ++i)
    {
        REG_DMA3SAD = (u32)&tests_array[0];
        REG_DMA3DAD = (u32)iwram_buf;
        REG_DMA3CNT = (32/4) | DMA_ENABLE | DMA_32BIT | DMA_SRC_INC | DMA_DST_INC;
        while (REG_DMA3CNT & DMA_ENABLE);

        for (int j = 32; j < 32768; j += 32)
        {
            REG_DMA3SAD = (u32)&tests_array[j];
            REG_DMA3DAD = (u32)ewram_buf;
            REG_DMA3CNT = (32/4) | DMA_ENABLE | DMA_32BIT | DMA_SRC_INC | DMA_DST_INC;

            process_data_chunk(iwram_buf, 32, j - 32, &total_result);

            while (REG_DMA3CNT & DMA_ENABLE);

            REG_DMA3SAD = (u32)ewram_buf;
            REG_DMA3DAD = (u32)iwram_buf;
            REG_DMA3CNT = (32/4) | DMA_ENABLE | DMA_32BIT | DMA_SRC_INC | DMA_DST_INC;
            while (REG_DMA3CNT & DMA_ENABLE);
        }

        process_data_chunk(iwram_buf, 32, 32768 - 32, &total_result);
    }

    global_result += total_result;
    return total_result & 0xFF;
}

// Test 9: Double-buffered with two IWRAM buffers (32-byte chunks)
static int8_t dbuf_a[32] IN_IWRAM __attribute__((aligned(4)));
static int8_t dbuf_b[32] IN_IWRAM __attribute__((aligned(4)));

static u32 __attribute__((section(".iwram_text"), noinline))
test_load_from_rom_to_iwram_with_dma_double_iwram_timed()
{
    volatile u32 total_result = 0;

    for (int pass = 0; pass < NB_ACCESS; ++pass)
    {
        REG_DMA3SAD = (u32)&tests_array[0];
        REG_DMA3DAD = (u32)dbuf_a;
        REG_DMA3CNT = (32 / 4) | DMA_ENABLE | DMA_32BIT | DMA_SRC_INC | DMA_DST_INC;
        while (REG_DMA3CNT & DMA_ENABLE);

        int8_t* current = dbuf_a;
        int8_t* next = dbuf_b;

        for (int off = 32; off < 32768; off += 32)
        {
            REG_DMA1SAD = (u32)&tests_array[off];
            REG_DMA1DAD = (u32)next;
            REG_DMA1CNT = (32 / 4) | DMA_ENABLE | DMA_32BIT | DMA_SRC_INC | DMA_DST_INC;

            process_data_chunk(current, 32, off - 32, &total_result);

            while (REG_DMA1CNT & DMA_ENABLE);

            int8_t* tmp = current;
            current = next;
            next = tmp;
        }

        process_data_chunk(current, 32, 32768 - 32, &total_result);
    }

    global_result += total_result;
    return total_result & 0xFF;
}

// Test 10: Optimized double buffering (512-byte chunks)
#define CHUNK_SIZE 512
static int8_t opt_buf_a[CHUNK_SIZE] IN_IWRAM __attribute__((aligned(4)));
static int8_t opt_buf_b[CHUNK_SIZE] IN_IWRAM __attribute__((aligned(4)));

static u32 __attribute__((section(".iwram_text"), noinline))
test_optimized_double_buffering()
{
    volatile u32 total_result = 0;
    int8_t* current = opt_buf_a;
    int8_t* next = opt_buf_b;

    REG_DMA3SAD = (u32)&tests_array[0];
    REG_DMA3DAD = (u32)current;
    REG_DMA3CNT = (CHUNK_SIZE / 4) | DMA_ENABLE | DMA_32BIT | DMA_SRC_INC | DMA_DST_INC;
    while (REG_DMA3CNT & DMA_ENABLE);

    for (int off = CHUNK_SIZE; off < 32768; off += CHUNK_SIZE)
    {
        REG_DMA1SAD = (u32)&tests_array[off];
        REG_DMA1DAD = (u32)next;
        REG_DMA1CNT = (CHUNK_SIZE / 4) | DMA_ENABLE | DMA_32BIT | DMA_SRC_INC | DMA_DST_INC;

        process_data_chunk(current, CHUNK_SIZE, off - CHUNK_SIZE, &total_result);

        while (REG_DMA1CNT & DMA_ENABLE);

        int8_t* tmp = current;
        current = next;
        next = tmp;
    }

    process_data_chunk(current, CHUNK_SIZE, 32768 - CHUNK_SIZE, &total_result);

    global_result += total_result;
    return total_result & 0xFF;
}

IN_EWRAM int ewram_var = 1234;
IN_IWRAM int iwram_var = 5678;

// All tests
static inline int all_tests()
{
    u32 start_time, end_time, result;
    u32 best_cycles = 0xFFFFFFFF;
    const char *best_name = "none";
    u32 best_result = 0;

    iprintf("Running GBA memory transfer benchmarks...\n");
    iprintf("ewram_var: %p\n", &ewram_var);
    iprintf("iwram_var: %p\n", &iwram_var);

    start_time = timer_start(3);
    result = test_load_from_rom();
    end_time = timer_stop(3);
    iprintf("Direct ROM : %lu cycles (result: %lu)\n", end_time, result);
    if (end_time < best_cycles) { best_cycles = end_time; best_name = "Direct ROM"; best_result = result; }

    start_time = timer_start(3);
    result = test_load_from_rom_with_cpuFastSet();
    end_time = timer_stop(3);
    iprintf("CpuFastSet : %lu cycles (result: %lu)\n", end_time, result);
    if (end_time < best_cycles) { best_cycles = end_time; best_name = "CpuFastSet"; best_result = result; }

    start_time = timer_start(3);
    result = test_load_from_rom_to_iwram_with_dma();
    end_time = timer_stop(3);
    iprintf("DMA Single : %lu cycles (result: %lu)\n", end_time, result);
    if (end_time < best_cycles) { best_cycles = end_time; best_name = "DMA Single"; best_result = result; }

    start_time = timer_start(3);
    result = test_double_buffered_dma();
    end_time = timer_stop(3);
    iprintf("Dual DMA   : %lu cycles (result: %lu)\n", end_time, result);
    if (end_time < best_cycles) { best_cycles = end_time; best_name = "Dual DMA"; best_result = result; }

    start_time = timer_start(3);
    result = test_realistic_double_buffering();
    end_time = timer_stop(3);
    iprintf("Realistic  : %lu cycles (result: %lu)\n", end_time, result);
    if (end_time < best_cycles) { best_cycles = end_time; best_name = "Realistic"; best_result = result; }

    start_time = timer_start(3);
    result = test_double_buffered_dma3_only();
    end_time = timer_stop(3);
    iprintf("DMA3 Only  : %lu cycles (result: %lu)\n", end_time, result);
    if (end_time < best_cycles) { best_cycles = end_time; best_name = "DMA3 Only"; best_result = result; }

    start_time = timer_start(3);
    result = test_load_from_rom_to_iwram_with_dma_double_iwram_timed();
    end_time = timer_stop(3);
    iprintf("Double IWRAM: %lu cycles (result: %lu)\n", end_time, result);
    if (end_time < best_cycles) { best_cycles = end_time; best_name = "Double IWRAM"; best_result = result; }

    start_time = timer_start(3);
    result = test_optimized_double_buffering();
    end_time = timer_stop(3);
    iprintf("Optimized  : %lu cycles (result: %lu)\n", end_time, result);
    if (end_time < best_cycles) { best_cycles = end_time; best_name = "Optimized"; best_result = result; }

    iprintf("\nGlobal result: %lu\n", global_result);
    iprintf("Best strategy: %s -> %lu cycles (result: %lu)\n", best_name, best_cycles, best_result);

    return 0;
}

#endif // TESTS_H
