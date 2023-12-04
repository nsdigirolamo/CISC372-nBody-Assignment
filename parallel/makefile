CC = nvcc
SDIR = sources
IDIR = headers
ODIR = objects

OBJS += $(ODIR)/compute.o
OBJS += $(ODIR)/kernel_utils.o
OBJS += $(ODIR)/memory_utils.o
OBJS += $(ODIR)/nbody.o

HDRS += $(IDIR)/compute.cuh
HDRS += $(IDIR)/config.cuh
HDRS += $(IDIR)/kernel_utils.cuh
HDRS += $(IDIR)/memory_utils.cuh
HDRS += $(IDIR)/nbody.cuh
HDRS += $(IDIR)/planets.cuh
HDRS += $(IDIR)/vector.cuh

CFLAGS = -I $(IDIR) -DDEBUG
LIBS = -lm
ALWAYS_REBUILD = makefile

run: nbody
	srun ./nbody > out.txt

nbody: $(OBJS)
	$(CC) $(CFLAGS) $^ -o $@ $(LIBS)

$(ODIR)/%.o: $(SDIR)/%.cu $(HDRS) $(ALWAYS_REBUILD) | $(ODIR) 
	$(CC) $(CFLAGS) -c $< -o $@

$(ODIR):
	mkdir -p $@

clean:
	rm -f $(OBJS) nbody out.txt
