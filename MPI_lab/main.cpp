#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <mpi.h>

int ProcNum = 0;  // Кількість доступних процесів
int ProcRank = 0; // Ранг поточного процесу

const int STEPS = 100; 


void ProcessInitialization(int &Size, int &LocalSize) {
    int RestSims; 
    int i;
    
    setvbuf(stdout, 0, _IONBF, 0);

    if (ProcRank == 0) {
        do {
            printf("\n--- Monte-Carlo: 1D Random Walk ---\n");
            printf("Enter total number of simulations (Size): ");
            scanf("%d", &Size);
            if (Size < ProcNum) {
                printf("Size must be greater than number of processes!\n");
            }
        } while (Size < ProcNum);
    }
    
    // Розсилаємо загальний розмір Size всім процесам
    MPI_Bcast(&Size, 1, MPI_INT, 0, MPI_COMM_WORLD);
    
    // Алгоритм чесного розподілу ітерацій з Task 11 методички
    RestSims = Size;
    for (i = 0; i < ProcRank; i++) {
        RestSims = RestSims - RestSims / (ProcNum - i);
    }
    LocalSize = RestSims / (ProcNum - ProcRank);
    
    // Ініціалізуємо генератор унікальним ключем для КОЖНОГО процесу
    srand((unsigned)time(NULL) + ProcRank * 1000);
}


void SerialResultCalculation(int Size, double &SerialResult) {
    double total_sq_dist = 0.0;
    
    for (int i = 0; i < Size; i++) {
        int position = 0;
        // Симулюємо один випадковий процес (100 кроків)
        for (int s = 0; s < STEPS; s++) {
            if (rand() % 2 == 0) position++;
            else position--;
        }
        total_sq_dist += (position * position);
    }
    SerialResult = total_sq_dist / Size;
}


void ParallelResultCalculation(int LocalSize, double &LocalSum) {
    LocalSum = 0.0;
    
    for (int i = 0; i < LocalSize; i++) {
        int position = 0;
        // Симулюємо один випадковий процес
        for (int s = 0; s < STEPS; s++) {
            if (rand() % 2 == 0) position++;
            else position--;
        }
        LocalSum += (position * position);
    }
}


void ResultReplication(double LocalSum, double &ParallelResult, int Size) {
    double GlobalSum = 0.0;
    
    // Збираємо локальні суми з усіх процесів у глобальну суму
    MPI_Reduce(&LocalSum, &GlobalSum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    
    if (ProcRank == 0) {
        ParallelResult = GlobalSum / Size;
    }
}


void TestResult(int Size, double ParallelResult) {
    double SerialResult;
    
    if (ProcRank == 0) {
        SerialResultCalculation(Size, SerialResult);
        
        printf("\n=== Verification ===\n");
        printf("Theoretical Expected Result : %.2f\n", (double)STEPS);
        printf("Parallel Result (MPI)       : %.2f\n", ParallelResult);
        printf("Serial Result (1 Core)      : %.2f\n", SerialResult);
        
        // Перевірка похибки 
        double diff = fabs(ParallelResult - SerialResult);
        if (diff > 0.02 * SerialResult) {
            printf("\nCONCLUSION: The results differ significantly. Check your code.\n");
        } else {
            printf("\nCONCLUSION: The results are practically identical (within statistical error).\n");
        }
    }
}


int main(int argc, char* argv[]) {
    int Size;                 
    int LocalSize;            
    double LocalSum;          
    double ParallelResult;    
    
    double Start, Finish, Duration; 

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &ProcNum);
    MPI_Comm_rank(MPI_COMM_WORLD, &ProcRank);

    ProcessInitialization(Size, LocalSize);

    Start = MPI_Wtime();

    ParallelResultCalculation(LocalSize, LocalSum);

    ResultReplication(LocalSum, ParallelResult, Size);

    Finish = MPI_Wtime();
    Duration = Finish - Start;

    TestResult(Size, ParallelResult);

    if (ProcRank == 0) {
        printf("\nTime of parallel execution = %f seconds\n", Duration);
        printf("------------------------------------\n");
    }

    MPI_Finalize();
    return 0;
}
