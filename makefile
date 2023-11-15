# https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html#examples

FLAGS=
LIBS= -lm
ALWAYS_REBUILD=makefile

pnbody: nbody.o pcompute.o
	nvcc $(FLAGS) $^ -o $@ $(LIBS)
pcompute.o: pcompute.cu config.h vector.h $(ALWAYS_REBUILD)
	nvcc $(FLAGS) -c $<

nbody: nbody.o compute.o
	nvcc $(FLAGS) $^ -o $@ $(LIBS)
nbody.o: nbody.c planets.h config.h vector.h $(ALWAYS_REBUILD)
	nvcc $(FLAGS) -c $< 
compute.o: compute.c config.h vector.h $(ALWAYS_REBUILD)
	nvcc $(FLAGS) -c $<

clean:
	rm -f *.o nbody pnbody output.txt poutput.txt
