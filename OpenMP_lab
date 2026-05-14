#include <stdio.h>
#include <stdlib.h>
#include <omp.h>  
#include <math.h>
#include <time.h>

const int STEPS = 100; // Кількість кроків однієї частинки

int main() {
    int Size;
    printf("\n--- Monte-Carlo: 1D Random Walk (OpenMP) ---\n");
    printf("Enter total number of simulations (Size): ");
    scanf("%d", &Size);

    double Start, Finish, DurationParallel, DurationSerial;
    double SerialResult = 0.0, ParallelResult = 0.0;

    // послідовний алгоритм (На 1 ядрі - для перевірки)
    Start = omp_get_wtime();
    double serial_total_sq_dist = 0.0;
    unsigned int serial_seed = (unsigned)time(NULL);

    for (int i = 0; i < Size; i++) {
        int position = 0;
        for (int s = 0; s < STEPS; s++) {
            // Швидкий потокобезпечний генератор випадкових чисел
            serial_seed = (1103515245 * serial_seed + 12345);
            if (((serial_seed / 65536) % 32768) % 2 == 0) position++;
            else position--;
        }
        serial_total_sq_dist += (position * position);
    }
    SerialResult = serial_total_sq_dist / Size;
    Finish = omp_get_wtime();
    DurationSerial = Finish - Start;


    // 2. паралельний алгоритм (OpenMP)
    double parallel_total_sq_dist = 0.0;
    int num_threads = 0;

    Start = omp_get_wtime();

    // розділити цикл for між ядрами і зібрати суму 
    #pragma omp parallel reduction(+:parallel_total_sq_dist)
    {
        // Кожен потік отримує свій унікальний seed для випадкових чисел
        unsigned int thread_seed = (unsigned)time(NULL) ^ omp_get_thread_num();

        // Записуємо, скільки ядер задіяно
        #pragma omp single
        num_threads = omp_get_num_threads();

        #pragma omp for
        for (int i = 0; i < Size; i++) {
            int position = 0;
            for (int s = 0; s < STEPS; s++) {
                thread_seed = (1103515245 * thread_seed + 12345);
                if (((thread_seed / 65536) % 32768) % 2 == 0) position++;
                else position--;
            }
            parallel_total_sq_dist += (position * position);
        }
    } 
    
    ParallelResult = parallel_total_sq_dist / Size;
    Finish = omp_get_wtime();
    DurationParallel = Finish - Start;


    // вивід результатів та аналіз швидкодії
    printf("\n=== Verification & Performance ===\n");
    printf("Threads used: %d\n", num_threads);
    printf("Theoretical Expected Result      : %.2f\n", (double)STEPS);
    printf("Serial Result (1 Core)           : %.2f\n", SerialResult);
    printf("Parallel Result (OpenMP)         : %.2f\n", ParallelResult);

    double diff = fabs(ParallelResult - SerialResult);
    if (diff > 0.02 * SerialResult) {
        printf("\nCONCLUSION: The results differ significantly.\n");
    } else {
        printf("\nCONCLUSION: The results are practically identical (Math is correct!).\n");
    }

    printf("\n--- Timings ---\n");
    printf("Serial Execution Time   : %f seconds\n", DurationSerial);
    printf("Parallel Execution Time : %f seconds\n", DurationParallel);
    printf("Speedup: %.2fx\n", DurationSerial / DurationParallel);
    printf("------------------------------------\n");

    return 0;
}
