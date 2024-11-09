# Communication Component

## Build

```bash
/opt/rocm-6.2.1/bin/hipcc -std=c++17 --offload-arch=gfx90a \
-DENABLE_MPI \
-I /home1/student35/liangyu/mscclpp/include \
-I /opt/ohpc/pub/mpi/openmpi4-gnu12/4.1.5/include \
-L /home1/student35/liangyu/mscclpp/build -lmscclpp \
-L /opt/ohpc/pub/mpi/openmpi4-gnu12/4.1.5/lib -lmpi \
comm.cu comm_test.cpp -o comm_test
```

## Run

```bash
mpirun -np 8 -x LD_LIBRARY_PATH=/home1/student35/liangyu/mscclpp/build:$LD_LIBRARY_PATH ./comm_test
```