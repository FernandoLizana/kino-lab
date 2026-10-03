#include <stdio.h>
#include <stdlib.h>
#include <time.h>

void generar_simulaciones(int num_simulaciones, int num_numeros, int num_selecciones, int *resultado) {
    srand((unsigned int)time(NULL)); // Inicializa la semilla para la generación de números aleatorios

    for (int i = 0; i < num_simulaciones; i++) {
        for (int j = 0; j < num_selecciones; j++) {
            resultado[i * num_selecciones + j] = rand() % num_numeros + 1;
        }
    }
}
