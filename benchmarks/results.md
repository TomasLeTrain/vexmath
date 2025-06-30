All of the following results were obtained by running the respective tests on a v5 brain

Latest benchmarks were taken `June 30 2025`

# Results
## Xoroshiro benchmarks
The multiple float benchmarks might be questionable since the compiler might be optimizing the elements that get overwritten

```
---------------------
running xoroshiro benchmarks
benching                            uniform float .. -> 50000 elements in 1.77 milliseconds -> 28.28 numbers/microsecond
benching                              uniform int .. -> 50000 elements in 0.84 milliseconds -> 59.78 numbers/microsecond
benching                     vector uniform float .. -> 50000 elements in 0.69 milliseconds -> 72.25 numbers/microsecond
benching          vector uniform doubleNext float .. -> 50000 elements in 0.53 milliseconds -> 94.53 numbers/microsecond
benching                       vector uniform int .. -> 50000 elements in 0.73 milliseconds -> 68.44 numbers/microsecond
benching               vector diff_float multiple .. -> 150000 elements in 1.73 milliseconds -> 86.69 numbers/microsecond
benching                    vector diff_float one .. -> 150000 elements in 1.62 milliseconds -> 92.87 numbers/microsecond
benching        vector diff_float multiple double .. -> 150000 elements in 1.65 milliseconds -> 90.71 numbers/microsecond
benching             vector diff_float one double .. -> 150000 elements in 1.65 milliseconds -> 90.73 numbers/microsecond
---------------------
```

## Sin/Cos taylor approximation benchmarks

```
---------------------
running taylor benchmarks
benching               taylor .. -> 100000 elements in 4.97 milliseconds -> 20.11 numbers/microsecond
benching         taylor_delta .. -> 100000 elements in 4.34 milliseconds -> 23.07 numbers/microsecond
benching              Vtaylor .. -> 100000 elements in 3.32 milliseconds -> 30.11 numbers/microsecond
benching        Vtaylor_delta .. -> 100000 elements in 3.09 milliseconds -> 32.36 numbers/microsecond
---------------------
```

## Sqrt approximation error
The sqrt test uses random numbers to determine the error bound so its possible the actual error bound is higher (however it should be close to this).
```
testing square root
max difference: 0.0179558
```

# Old benchmarks
<details>
<summary>June 25 2025</summary>

## Xoroshiro benchmarks

```
---------------------
running xoroshiro benchmarks
benching                            uniform float .. -> 50000 elements in 1.77 milliseconds -> 28.28 numbers/microsecond
benching                              uniform int .. -> 50000 elements in 0.84 milliseconds -> 59.31 numbers/microsecond
benching                     vector uniform float .. -> 50000 elements in 0.69 milliseconds -> 72.22 numbers/microsecond
benching          vector uniform doubleNext float .. -> 50000 elements in 0.53 milliseconds -> 94.89 numbers/microsecond
benching                       vector uniform int .. -> 50000 elements in 0.73 milliseconds -> 68.46 numbers/microsecond
benching               vector diff_float multiple .. -> 50000 elements in 1.73 milliseconds -> 28.90 numbers/microsecond
benching                    vector diff_float one .. -> 50000 elements in 1.62 milliseconds -> 30.96 numbers/microsecond
benching        vector diff_float multiple double .. -> 50000 elements in 1.65 milliseconds -> 30.23 numbers/microsecond
benching             vector diff_float one double .. -> 50000 elements in 1.65 milliseconds -> 30.25 numbers/microsecond
---------------------
```

## Sin/Cos taylor approximation benchmarks

```
---------------------
running taylor benchmarks
benching               taylor .. -> 100000 elements in 4.34 milliseconds -> 23.05 numbers/microsecond
benching         taylor_delta .. -> 100000 elements in 3.94 milliseconds -> 25.36 numbers/microsecond
benching              Vtaylor .. -> 100000 elements in 3.32 milliseconds -> 30.10 numbers/microsecond
benching        Vtaylor_delta .. -> 100000 elements in 3.09 milliseconds -> 32.36 numbers/microsecond
---------------------
```

## Sqrt approximation error

```
testing square root
max difference: 0.0177002
```

</details>
