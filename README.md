Vygenerujte překladový skript pomocí cmake a spusťte překlad:

```
cmake -Bbuild -S.
```

```
cmake --build build --config Release
```

```
mpiexec -np 2 ./bin {param}
```
