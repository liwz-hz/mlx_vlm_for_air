#include <arm_neon.h>
#include <pthread.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define TILE 16
#define MAX_T 8

typedef struct {
    int t_len;
    int k;
    int n_groups;
    int n_rows;
    const float *x;
    const uint32_t *wq;
    const float *scales;
    const float *biases;
    const float *gsums;
    float *out;
    int row_start;
    int row_end;
} job_t;

static const int32x4_t SH_LO = {0, -4, -8, -12};
static const int32x4_t SH_HI = {-16, -20, -24, -28};
static const uint32x4_t MASK4 = {0xF, 0xF, 0xF, 0xF};

static void *worker(void *arg) {
    job_t *j = (job_t *)arg;
    const int T = j->t_len;
    const int K = j->k;
    const int G = j->n_groups;
    const int k8 = K / 8;
    float xs[MAX_T][64] __attribute__((aligned(16)));

    for (int r0 = j->row_start; r0 < j->row_end; r0 += TILE) {
        int nr = j->row_end - r0 < TILE ? j->row_end - r0 : TILE;
        for (int t = 0; t < T; t++) {
            for (int ri = 0; ri < nr; ri++) {
                j->out[(size_t)t * j->n_rows + r0 + ri] = 0.0f;
            }
        }
        for (int g = 0; g < G; g++) {
            for (int t = 0; t < T; t++) {
                const float *xg = j->x + (size_t)t * K + (size_t)g * 64;
                memcpy(xs[t], xg, sizeof(float) * 64);
            }
            for (int ri = 0; ri < nr; ri++) {
                int row = r0 + ri;
                const uint32_t *wp = j->wq + (size_t)row * k8 + (size_t)g * 8;
                const float sc = j->scales[(size_t)row * G + g];
                const float bi = j->biases[(size_t)row * G + g];
                for (int t = 0; t < T; t++) {
                    const float *xt = xs[t];
                    float32x4_t a0 = vdupq_n_f32(0.0f);
                    float32x4_t a1 = vdupq_n_f32(0.0f);
                    for (int c = 0; c < 8; c += 2) {
                        uint32x4_t v0 = vdupq_n_u32(wp[c]);
                        uint32x4_t v1 = vdupq_n_u32(wp[c + 1]);
                        float32x4_t n0 = vcvtq_f32_u32(
                            vandq_u32(vshlq_u32(v0, SH_LO), MASK4));
                        float32x4_t n1 = vcvtq_f32_u32(
                            vandq_u32(vshlq_u32(v0, SH_HI), MASK4));
                        float32x4_t n2 = vcvtq_f32_u32(
                            vandq_u32(vshlq_u32(v1, SH_LO), MASK4));
                        float32x4_t n3 = vcvtq_f32_u32(
                            vandq_u32(vshlq_u32(v1, SH_HI), MASK4));
                        a0 = vfmaq_f32(a0, n0, vld1q_f32(xt + c * 8));
                        a1 = vfmaq_f32(a1, n1, vld1q_f32(xt + c * 8 + 4));
                        a0 = vfmaq_f32(a0, n2, vld1q_f32(xt + c * 8 + 8));
                        a1 = vfmaq_f32(a1, n3, vld1q_f32(xt + c * 8 + 12));
                    }
                    float e0[4], e1[4];
                    vst1q_f32(e0, a0);
                    vst1q_f32(e1, a1);
                    float ga = (e0[0] + e0[1]) + (e0[2] + e0[3]) +
                               (e1[0] + e1[1]) + (e1[2] + e1[3]);
                    j->out[(size_t)t * j->n_rows + row] +=
                        sc * ga + bi * j->gsums[(size_t)t * G + g];
                }
            }
        }
    }
    return NULL;
}

void cpu_qmv(int t_len, int k, int n_rows,
             const float *x, const uint32_t *wq,
             const float *scales, const float *biases,
             float *out, int n_threads) {
    const int G = k / 64;
    float *gsums = (float *)malloc(sizeof(float) * (size_t)t_len * G);
    for (int t = 0; t < t_len; t++) {
        const float *xr = x + (size_t)t * k;
        for (int g = 0; g < G; g++) {
            const float *xg = xr + (size_t)g * 64;
            float32x4_t s0 = vdupq_n_f32(0.0f);
            float32x4_t s1 = vdupq_n_f32(0.0f);
            for (int c = 0; c < 8; c++) {
                s0 = vaddq_f32(s0, vld1q_f32(xg + c * 8));
                s1 = vaddq_f32(s1, vld1q_f32(xg + c * 8 + 4));
            }
            float b0[4], b1[4];
            vst1q_f32(b0, s0);
            vst1q_f32(b1, s1);
            gsums[(size_t)t * G + g] =
                (b0[0] + b0[1]) + (b0[2] + b0[3]) +
                (b1[0] + b1[1]) + (b1[2] + b1[3]);
        }
    }

    if (n_threads < 1) n_threads = 1;
    if (n_threads > n_rows) n_threads = n_rows;
    job_t *jobs = (job_t *)malloc(sizeof(job_t) * n_threads);
    pthread_t *tid = (pthread_t *)malloc(sizeof(pthread_t) * n_threads);
    int tile_chunks = (n_rows + TILE - 1) / TILE;
    if (n_threads > tile_chunks) n_threads = tile_chunks;
    int base = tile_chunks / n_threads;
    int rem = tile_chunks % n_threads;
    int start = 0;
    int spawned = 0;
    for (int i = 0; i < n_threads; i++) {
        int chunks = base + (i < rem ? 1 : 0);
        if (chunks <= 0) continue;
        int end = start + chunks * TILE;
        if (end > n_rows) end = n_rows;
        jobs[spawned] = (job_t){t_len, k, G, n_rows, x, wq, scales,
                                biases, gsums, out, start, end};
        pthread_create(&tid[spawned], NULL, worker, &jobs[spawned]);
        spawned++;
        start = end;
    }
    for (int i = 0; i < spawned; i++) pthread_join(tid[i], NULL);
    free(tid);
    free(jobs);
    free(gsums);
}
