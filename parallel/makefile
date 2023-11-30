FLAGS = -DDEBUG -DALT_PRINT_SYSTEM
LIBS = -lm
ALWAYS_REBUILD = makefile

run: nbody
	srun ./nbody > out.txt

nbody: nbody.o compute.o
	nvcc $(FLAGS) $^ -o $@ $(LIBS)
nbody.o: nbody.cu planets.h config.h vector.h $(ALWAYS_REBUILD)
	nvcc $(FLAGS) -c $<
compute.o: compute.cu config.h vector.h $(ALWAYS_REBUILD)
	nvcc $(FLAGS) -c $<

clean:
	rm -f *.o nbody pnbody out.txt pout.txt
