(
    cd serial; 
    echo "Compiling serial..."; 
    make; 
    echo "Running serial..."; 
    ./nbody > out.txt; 
    echo "Serial complete!"
)
(
    cd parallel; 
    echo "Compiling parallel..."; 
    make; echo "Running parallel..."; 
    srun ./nbody > out.txt; 
    echo "Parallel complete!"
)