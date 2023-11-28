cd parallel;
echo "Compiling parallel...";
make;
echo "Removing previous output...";
rm -f marathon.txt;
echo "Running marathon...";
for i in {1..5}; do
    srun ./nbody >> marathon.txt; 
done;
echo "Marathon complete! Printing results...";
sed 's/[^0-9.]//g' marathon.txt;
