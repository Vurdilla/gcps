#ifndef VC3_PHYS_KSNP_KERNEL_TIMERASYNC
#define VC3_PHYS_KSNP_KERNEL_TIMERASYNC

#include <cuda_runtime.h>
#include <unordered_map>
#include <string>
#include <iostream>
#include <iomanip>
#include <mutex>
#include <vector>
#include <map>
#include <numeric>

#include "../include/cuda_error_hadling.h"

struct kernelStatsAsync {
    float total_ms = 0.0f;
    int launches = 0;
};

class kernelTimerAsync {
private:
    std::unordered_map<std::string, kernelStatsAsync> stats;
    mutable std::mutex mtx;

    struct EventRecordAsync {
        std::string name;
        cudaEvent_t start, stop;
    };

    // Per-thread storage to avoid race conditions on the vector
    static thread_local std::vector<EventRecordAsync> records_for_step;
    static thread_local cudaEvent_t step_start_event;

    float total_gpu_time_ms_accumulated = 0.0f;
    float total_shadow_time_ms_accumulated = 0.0f;
    int steps_recorded = 0;

    // --- PRIVATE CONSTRUCTOR ---
    // This prevents direct creation of objects.
    kernelTimerAsync() = default;

public:
    // --- DELETED FUNCTIONS ---
    // This prevents copying, ensuring only one instance can ever exist.
    kernelTimerAsync(const kernelTimerAsync&) = delete;
    kernelTimerAsync& operator=(const kernelTimerAsync&) = delete;

    // --- PUBLIC STATIC ACCESSOR (THE KEY TO THE FIX) ---
    // This function is the ONLY way to get the timer.
    // It creates the instance once and only once, in a thread-safe way.
    static kernelTimerAsync& getInstance() {
        static kernelTimerAsync instance;
        return instance;
    }

    // The destructor is now safe because it will only be called once at program exit.
    ~kernelTimerAsync() {
        if (step_start_event) {
            // We should check if the event is valid before destroying.
            // But since end_step cleans up, this is mostly for the main thread's event.
        }
    }

    template<typename KernelFunc, typename T1>
    void record(const std::string& name, KernelFunc kernel, dim3 gridDim, dim3 blockDim, size_t sharedMem, cudaStream_t stream, T1 arg1)
    {
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start, stream);
        SAFE_KERNEL_CALL((kernel<<<gridDim, blockDim, sharedMem, stream>>>(arg1)));
        cudaEventRecord(stop, stream);
        records_for_step.push_back({ name, start, stop });
    }

    template<typename KernelFunc, typename T1, typename T2>
    void record(const std::string& name, KernelFunc kernel, dim3 gridDim, dim3 blockDim, size_t sharedMem, cudaStream_t stream, T1 arg1, T2 arg2)
    {
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start, stream);
        SAFE_KERNEL_CALL((kernel<<<gridDim, blockDim, sharedMem, stream>>>(arg1, arg2)));
        cudaEventRecord(stop, stream);
        records_for_step.push_back({ name, start, stop });
    }

    template<typename KernelFunc, typename T1, typename T2, typename T3>
    void record(const std::string& name, KernelFunc kernel, dim3 gridDim, dim3 blockDim, size_t sharedMem, cudaStream_t stream, T1 arg1, T2 arg2, T3 arg3)
    {
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start, stream);
        SAFE_KERNEL_CALL((kernel<<<gridDim, blockDim, sharedMem, stream>>>(arg1, arg2, arg3)));
        cudaEventRecord(stop, stream);
        records_for_step.push_back({ name, start, stop });
    }

    template<typename KernelFunc, typename T1, typename T2, typename T3, typename T4>
    void record(const std::string& name, KernelFunc kernel, dim3 gridDim, dim3 blockDim, size_t sharedMem, cudaStream_t stream, T1 arg1, T2 arg2, T3 arg3, T4 arg4)
    {
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start, stream);
        SAFE_KERNEL_CALL((kernel<<<gridDim, blockDim, sharedMem, stream>>>(arg1, arg2, arg3, arg4)));
        cudaEventRecord(stop, stream);
        records_for_step.push_back({ name, start, stop });
    }

    void start_step(cudaStream_t stream = 0) {
        if (!step_start_event) {
            cudaEventCreate(&step_start_event);
        }
        cudaEventRecord(step_start_event, stream);
    }

    void end_step() {
        if (records_for_step.empty()) return;

        cudaEventSynchronize(records_for_step.back().stop);

        float step_total_ms;
        cudaEventElapsedTime(&step_total_ms, step_start_event, records_for_step.back().stop);

        float step_kernel_sum_ms = 0.0f;

        for (const auto& rec : records_for_step) {
            float kernel_ms;
            cudaEventElapsedTime(&kernel_ms, rec.start, rec.stop);

            // This lock is essential because multiple threads call end_step
            std::lock_guard<std::mutex> lock(mtx);
            stats[rec.name].total_ms += kernel_ms;
            stats[rec.name].launches += 1;

            step_kernel_sum_ms += kernel_ms;

            cudaEventDestroy(rec.start);
            cudaEventDestroy(rec.stop);
        }

        records_for_step.clear();

        std::lock_guard<std::mutex> lock(mtx);
        total_gpu_time_ms_accumulated += step_total_ms;
        total_shadow_time_ms_accumulated += (step_total_ms - step_kernel_sum_ms);
        steps_recorded += 1;
    }

    void print_summary(std::ostream& out = std::cout) const
    {
        // Lock the mutex to ensure thread-safe reading of the shared stats map,
        // preventing race conditions if another thread is still finishing its last step.
        std::lock_guard<std::mutex> lock(mtx);

        // --- Per-Kernel Breakdown ---
        out << "\n--- KERNEL TIMING SUMMARY (Tab-Separated) ---\n";
        out << "Kernel Name\t" << "Launches\t" << "Total (s)\t" << "Average (s)\n";

        // Use a temporary map to sort the results alphabetically by kernel name for readability.
        std::map<std::string, kernelStatsAsync> sorted_stats(stats.begin(), stats.end());

        for (const auto& kv : sorted_stats) {
            const auto& name = kv.first;
            const auto& s = kv.second;

            // Convert milliseconds to seconds for all calculations and output
            const double total_s = s.total_ms / 1000.0;
            const double avg_s = (s.launches > 0) ? (total_s / s.launches) : 0.0;

            out << name << "\t"
                << s.launches << "\t"
                << std::fixed << std::setprecision(6) // Use 6 decimal places for seconds
                << total_s << "\t"
                << avg_s << "\n";
        }

        // --- Global Aggregated Summary ---
        // Calculate the total kernel time by summing up the individual totals.
        double total_kernel_time_s = 0.0;
        for (const auto& kv : stats) {
            total_kernel_time_s += kv.second.total_ms;
        }
        total_kernel_time_s /= 1000.0;

        const double total_gpu_time_s = total_gpu_time_ms_accumulated / 1000.0;
        const double total_shadow_time_s = total_shadow_time_ms_accumulated / 1000.0;

        out << "\n--- GLOBAL TIMING SUMMARY ---\n";
        out << "Total Steps Recorded:\t" << steps_recorded << "\n";
        out << std::fixed << std::setprecision(6);
        out << "Total GPU Time:\t\t" << total_gpu_time_s << " s\n";
        out << "Total Kernel Time:\t" << total_kernel_time_s << " s\n";
        out << "Total Shadow Time:\t" << total_shadow_time_s << " s (Overhead)\n";

        if (steps_recorded > 0) {
            out << "Avg Total GPU Time/Step:\t" << total_gpu_time_s / steps_recorded << " s\n";
            out << "Avg Shadow Time/Step:\t" << total_shadow_time_s / steps_recorded << " s\n";
        }
        out << "-------------------------------------------\n";
    }
};

// --- DEFINE the thread_local static members ---
thread_local std::vector<kernelTimerAsync::EventRecordAsync> kernelTimerAsync::records_for_step;
thread_local cudaEvent_t kernelTimerAsync::step_start_event = nullptr;

#endif