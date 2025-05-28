## Пример работы

1. Инициализация и обучение:
```bash
./learning new 784 128 10 --train train.csv --test test.csv
```
2. Загрузка и дообучение:
```bash
./learning load model.bin 3 --train train.csv --test test.csv
```

## Требования

- Компилятор C (gcc)
- Стандартная библиотека C
- CBLAS

## Сборка проекта

```bash
gcc main.c dataset_io.c neuralnetwork.c -L. -lopenblas -o learning.exe
```
