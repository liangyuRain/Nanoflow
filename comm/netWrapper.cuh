#include <cstdio>
#include "comm.h"
#include <vector>
#include <algorithm>
#include <memory>
#include <cassert>
#include <span>
#include "tensor.cuh"
#include <assert.h>

#ifdef ENABLE_MPI
// use mpi to bcast the handles
#include <mpi.h>
#else
// use thread sync to get the handles
#include "networkManager.cuh"
#endif

#include <mscclpp/core.hpp>
#include <mscclpp/concurrency_device.hpp>
#include <mscclpp/sm_channel.hpp>
#include <mscclpp/sm_channel_device.hpp>
#include <mscclpp/proxy_channel.hpp>
#include <mscclpp/proxy_channel_device.hpp>

#define CUDA_CHECK(status)                                              \
  {                                                                     \
    cudaError_t error = status;                                         \
    if (error != cudaSuccess) {                                         \
      std::cerr << "Got bad cuda status: " << cudaGetErrorString(error) \
                << " at line: " << __LINE__ << __FILE__ << std::endl;               \
      exit(EXIT_FAILURE);                                               \
    }                                                                   \
  }

struct vortexWeight{
    half* ptr;
    int N;
    int K;
    size_t size() const { return (size_t)N * K; }
};

class NetWrapper
{
    public:
    using Element=half;
    Element* input = nullptr;
    Element* output = nullptr;
    pllmTensor<Element> pllm_tensor_input;
    pllmTensor<Element> pllm_tensor_output;
    size_t input_size;
    size_t output_size;
    size_t rank, nranks;
    bool isInitialized = false;
    cudaStream_t stream;

    std::vector<mscclpp::SmChannel> smChannels;
    std::vector<mscclpp::DeviceHandle<mscclpp::SmChannel>> smChannelHandles;

    NetWrapper(){}
    void checkassert(){
        assert(input_size % nranks == 0);
        assert(reinterpret_cast<uintptr_t>(input) % sizeof(int4) == 0);
        assert(reinterpret_cast<uintptr_t>(output) % sizeof(int4) == 0);
    }

    pllmTensor<Element> getInput() {
        return pllm_tensor_input;
    }
    pllmTensor<Element> getOutput() {
        return pllm_tensor_output;
    }
    int nblocks, nthreads;
    bool sync_mode;
    virtual NetWrapper& configRun(int nblocks, int nthreads, bool sync) {
        this->nblocks = nblocks;
        this->nthreads = nthreads;
        this->sync_mode = sync;
        return *this;
    }


protected:
	void init(int rank, int nranks,
			  Element* input, Element* output,
			  size_t input_size, size_t output_size) {
		this->isInitialized = true;
		this->rank = rank;
		this->nranks = nranks;
		this->input = input;
		this->input_size = input_size;
		this->output = output;
		this->output_size = output_size;
		checkassert();
	}
    virtual void work() {
        operator()(stream, nblocks, nthreads, sync_mode);
    }
	virtual void operator()(cudaStream_t stream, int nblocks, int nthreads, bool sync){}

    virtual void sync(cudaStream_t stream) {}

    void setupSmChannels(std::shared_ptr<mscclpp::Communicator> comm,
                         std::vector<std::shared_ptr<mscclpp::Connection>> connections,
                         mscclpp::DeviceHandle<mscclpp::SmChannel>** smChannelHandlesCuda,
                         Element* input, Element* output, size_t input_size, size_t output_size) {
        const mscclpp::TransportFlags allTransports = mscclpp::Transport::CudaIpc;
        mscclpp::RegisteredMemory inputBuffRegMem = comm->registerMemory(input, input_size * sizeof(Element), allTransports);
        mscclpp::RegisteredMemory outputBuffRegMem;
        if (input != output) outputBuffRegMem = comm->registerMemory(output, output_size * sizeof(Element), allTransports);

        std::vector<mscclpp::NonblockingFuture<mscclpp::RegisteredMemory>> remoteRegMemories;
        mscclpp::RegisteredMemory& localRegMemory = (input != output) ? outputBuffRegMem : inputBuffRegMem;

        for (size_t r = 0; r < nranks; ++r) {
            if (r == rank) continue;
            comm->sendMemoryOnSetup(localRegMemory, r, 0);
            auto remoteMemory = comm->recvMemoryOnSetup(r, 0);
            remoteRegMemories.push_back(remoteMemory);
        }
        comm->setup();

        std::vector<std::shared_ptr<mscclpp::SmDevice2DeviceSemaphore>> smSemaphores;
        for (size_t i = 0; i < connections.size(); ++i) {
            smSemaphores.emplace_back(std::make_shared<mscclpp::SmDevice2DeviceSemaphore>(*comm, connections[i]));
        }
        comm->setup();
        for (size_t i = 0; i < connections.size(); ++i) {
            smChannels.emplace_back(smSemaphores[i], remoteRegMemories[i].get(), inputBuffRegMem.data());
            smChannelHandles.emplace_back(mscclpp::deviceHandle(smChannels.back()));
        }
        comm->setup();

        assert(connections.size() == nranks - 1);
        CUDA_CHECK(cudaMalloc(smChannelHandlesCuda, (nranks - 1) * sizeof(mscclpp::DeviceHandle<mscclpp::SmChannel>)));
        CUDA_CHECK(cudaMemcpy(*smChannelHandlesCuda, &smChannelHandles[smChannelHandles.size() - (nranks - 1)],
                              (nranks - 1) * sizeof(mscclpp::DeviceHandle<mscclpp::SmChannel>), cudaMemcpyHostToDevice));
    }

    void setupSmChannels(std::shared_ptr<mscclpp::Communicator> comm,
                         std::vector<std::shared_ptr<mscclpp::Connection>> connections,
                         mscclpp::DeviceHandle<mscclpp::SmChannel>** smChannelHandlesCuda,
                         Element* buff, size_t buff_size) {
        setupSmChannels(comm, connections, smChannelHandlesCuda, buff, buff, buff_size, buff_size);
    }
};

class NetAllGather: public NetWrapper
{
    public:
    bool columnwise = false;
    mscclpp::DeviceSyncer* syncersCuda;
    mscclpp::DeviceHandle<mscclpp::SmChannel>* smChannelHandlesCuda;

    NetAllGather():NetWrapper(){
       
    }
    // Init AllGather using explicit input and output buffers.
    void init(std::shared_ptr<mscclpp::Communicator> comm,
                    std::vector<std::shared_ptr<mscclpp::Connection>> connections,
                    int rank,
                    int nranks,
                    pllmTensor<Element> input,
                    pllmTensor<Element> output) {
        assert(input.layout == output.layout);
        this->pllm_tensor_input = input;
        this->pllm_tensor_output = output;
        NetWrapper::init(rank, nranks, input.ptr, output.ptr, input.size(), output.size());
    
        std::vector<mscclpp::DeviceSyncer> syncers(nranks - 1);
        CUDA_CHECK(cudaMalloc(&syncersCuda, syncers.size() * sizeof(mscclpp::DeviceSyncer)));
        CUDA_CHECK(cudaMemcpy(syncersCuda, syncers.data(), syncers.size() * sizeof(mscclpp::DeviceSyncer), cudaMemcpyHostToDevice));

        setupSmChannels(comm,
                        connections,
                        &smChannelHandlesCuda,
                        input.ptr,
                        output.ptr,
                        input.size(),
                        output.size());
    }

    void init(std::shared_ptr<mscclpp::Communicator> comm,
                    std::vector<std::shared_ptr<mscclpp::Connection>> connections,
                    int rank,
                    int nranks,
                    pllmTensor<Element> buff) {
            init(comm, connections, rank, nranks, buff, buff);
    }

    ~NetAllGather(){
        CUDA_CHECK(cudaFree(syncersCuda));
        CUDA_CHECK(cudaFree(smChannelHandlesCuda));
    }

    NetAllGather& setColumnwise(bool columnwise = true) {
        this->columnwise = columnwise;
        return *this;
    }

    void operator()(cudaStream_t stream, int nblocks, int nthreads, bool sync) {
        const int nchannels = nranks - 1;
        if (!columnwise) {
            assert(this->pllm_tensor_input.dim1 == this->pllm_tensor_output.dim1);
            assert(this->pllm_tensor_input.dim2 == this->pllm_tensor_output.dim2);
            const uint64_t nelem_per_shard = input_size / nranks;
            const uint64_t local_offset = rank * nelem_per_shard;
            if (sync) {
                allgatherKernelEntryPoint<<<nblocks, nthreads, 0, stream>>>(
                    smChannelHandlesCuda, syncersCuda, nchannels, local_offset, nelem_per_shard, input, output);
            } else {
                allgatherKernelWithoutSync<<<nblocks, nthreads, 0, stream>>>(
                    smChannelHandlesCuda, syncersCuda, nchannels, local_offset, nelem_per_shard, input, output);
            }
        } else {
            assert(this->pllm_tensor_input.dim1 == this->pllm_tensor_output.dim1);
            assert(this->pllm_tensor_input.dim2 == this->pllm_tensor_output.dim2 / nranks);
            const uint64_t input_ncols = this->pllm_tensor_input.dim2;
            const uint64_t output_ncols = this->pllm_tensor_output.dim2;
            const uint64_t output_row_offset = rank * input_ncols;
            const uint64_t nrows = this->pllm_tensor_input.dim1;

            constexpr uint64_t n_half_per_int4 = sizeof(int4) / sizeof(half);
            assert(input_ncols % n_half_per_int4 == 0);
            assert(output_ncols % n_half_per_int4 == 0);
            assert(output_row_offset % n_half_per_int4 == 0);
            assert(reinterpret_cast<uintptr_t>(input) % sizeof(int4) == 0);
            assert(reinterpret_cast<uintptr_t>(output) % sizeof(int4) == 0);

            columnwiseAllgatherKernelEntryPoint<<<nblocks, nthreads, 0, stream>>>(
                smChannelHandlesCuda, syncersCuda, sync, nchannels,
                input_ncols, output_ncols, output_row_offset, nrows, input, output);
        }
    }
    void sync(cudaStream_t stream) override {
        const int nchannels = nranks - 1;
        syncDevices<<<1, nchannels, 0, stream>>>(smChannelHandlesCuda, nchannels);
    }

    // override copy constructor
    NetAllGather(const NetAllGather& other) = delete;

};

class NetReduceScatter: public NetWrapper
{
    public:
    mscclpp::DeviceSyncer* syncerCuda;
    mscclpp::DeviceHandle<mscclpp::SmChannel>* smInputBuffChannelHandlesCuda;

    NetReduceScatter():NetWrapper(){}
    ~NetReduceScatter(){
        CUDA_CHECK(cudaFree(syncerCuda));
        CUDA_CHECK(cudaFree(smInputBuffChannelHandlesCuda));
    }
    void init(std::shared_ptr<mscclpp::Communicator> comm,
                     std::vector<std::shared_ptr<mscclpp::Connection>> connections,
                     int rank, int nranks,
                     Element* input, Element* output, size_t input_size, size_t output_size){
        NetWrapper::init(rank, nranks, input, output, input_size, output_size);

        setupSmChannels(comm, connections, &smInputBuffChannelHandlesCuda, input, input_size);
        mscclpp::DeviceSyncer syncer = mscclpp::DeviceSyncer();
        CUDA_CHECK(cudaMalloc(&syncerCuda, sizeof(mscclpp::DeviceSyncer)));
        CUDA_CHECK(cudaMemcpy(syncerCuda, &syncer, sizeof(mscclpp::DeviceSyncer), cudaMemcpyHostToDevice));
}
    void operator()(cudaStream_t stream, int nblocks, int nthreads, bool sync) override {
        const uint64_t nelem_per_shard = input_size / nranks;
        if (sync) {
            reduceScatterKernelEntryPoint<<<nblocks, nthreads, 0, stream>>>(
                smInputBuffChannelHandlesCuda, syncerCuda,
                rank, nranks, nelem_per_shard, input, output);
        } else {
            reduceScatterKernelWithoutSync<<<nblocks, nthreads, 0, stream>>>(
                smInputBuffChannelHandlesCuda, syncerCuda,
                rank, nranks, nelem_per_shard, input, output);
        }
    }
    void sync(cudaStream_t stream) override {
        const int nchannels = nranks - 1;
        syncDevices<<<1, nchannels, 0, stream>>>(smInputBuffChannelHandlesCuda, nchannels);
    }
};

class NetAllReduce: public NetWrapper
{
    public:
    mscclpp::DeviceSyncer* syncersCuda;    
    mscclpp::DeviceSyncer* globalSyncerCuda;
    mscclpp::DeviceHandle<mscclpp::SmChannel>* smInputBuffChannelHandlesCuda;
    mscclpp::DeviceHandle<mscclpp::SmChannel>* smOutputBuffChannelHandlesCuda;

    NetAllReduce():NetWrapper(){}

	void init(std::shared_ptr<mscclpp::Communicator> comm,
			  std::vector<std::shared_ptr<mscclpp::Connection>> connections,
			  int rank,
			  int nranks,
			  pllmTensor<Element> input,
			  pllmTensor<Element> output) {

        assert(input.layout == output.layout);
        this->pllm_tensor_input = input;
        this->pllm_tensor_output = output;
		NetWrapper::init(rank, nranks, input.ptr, output.ptr, input.size(), output.size());

		setupSmChannels(
			comm, connections, &smInputBuffChannelHandlesCuda, input.ptr, input.size());
		setupSmChannels(
			comm, connections, &smOutputBuffChannelHandlesCuda, output.ptr, output.size());

		std::vector<mscclpp::DeviceSyncer> syncers(nranks - 1);
		CUDA_CHECK(cudaMalloc(&syncersCuda, syncers.size() * sizeof(mscclpp::DeviceSyncer)));
		CUDA_CHECK(cudaMemcpy(syncersCuda,
							  syncers.data(),
							  syncers.size() * sizeof(mscclpp::DeviceSyncer),
							  cudaMemcpyHostToDevice));

		mscclpp::DeviceSyncer syncer = mscclpp::DeviceSyncer();
		CUDA_CHECK(cudaMalloc(&globalSyncerCuda, sizeof(mscclpp::DeviceSyncer)));
		CUDA_CHECK(cudaMemcpy(
			globalSyncerCuda, &syncer, sizeof(mscclpp::DeviceSyncer), cudaMemcpyHostToDevice));
	}

	~NetAllReduce(){
        CUDA_CHECK(cudaFree(syncersCuda));
        CUDA_CHECK(cudaFree(globalSyncerCuda));
        CUDA_CHECK(cudaFree(smInputBuffChannelHandlesCuda));
        CUDA_CHECK(cudaFree(smOutputBuffChannelHandlesCuda));
    }
    void operator()(cudaStream_t stream, int nblocks, int nthreads, bool sync) override {
        const int nchannels = nranks - 1;
        const uint64_t nelem_per_shard = input_size / nranks;
        const uint64_t local_offset = rank * nelem_per_shard;
        if (sync) {
            allreduceKernelEntryPoint<<<nblocks, nthreads, 0, stream>>>(
                smInputBuffChannelHandlesCuda, smOutputBuffChannelHandlesCuda, syncersCuda, globalSyncerCuda,
                nchannels, local_offset, rank, nranks, nelem_per_shard, input, output);
        } else {
            allreduceKernelWithoutSync<<<nblocks, nthreads, 0, stream>>>(
                smInputBuffChannelHandlesCuda, smOutputBuffChannelHandlesCuda, syncersCuda, globalSyncerCuda,
                nchannels, local_offset, rank, nranks, nelem_per_shard, input, output);
        }
    }
    void sync(cudaStream_t stream) override {
        const int nchannels = nranks - 1;
        syncDevices<<<1, nchannels, 0, stream>>>(smInputBuffChannelHandlesCuda, nchannels);
    }
};

class NetAllReduceWithLN : public NetAllReduce
{
    public:
    pllmTensor<half> ln_weight;
    float epsilon;
    bool run_ln = true;
    pllmTensor<Element> output_before_ln;

    void init(std::shared_ptr<mscclpp::Communicator> comm,
			  std::vector<std::shared_ptr<mscclpp::Connection>> connections,
			  int rank,
			  int nranks,
			  pllmTensor<Element> input,
			  pllmTensor<Element> output,
              pllmTensor<Element> output_before_ln) {
        NetAllReduce::init(comm, connections, rank, nranks, input, output);
        this -> output_before_ln = output_before_ln;
    }

    void setEpsilon(float epsilon) {
        this->epsilon = epsilon;
    }

    bool setWeight(vortexWeight weight) {
        ln_weight = pllmTensor<half>(weight.ptr, weight.size());
        return true;
    }

    NetAllReduceWithLN& runLn(bool run_ln = true) {
        this -> run_ln = run_ln;
        return *this;
    }

    void operator()(cudaStream_t stream, int nblocks, int nthreads, bool sync) override {
        const int nchannels = nranks - 1;
        const uint64_t nelem_per_shard = input_size / nranks;
        const uint64_t local_offset = rank * nelem_per_shard;
        if (run_ln){
            // spdlog::info("row {}, col {}", pllm_tensor_input.dim1/nranks, pllm_tensor_input.dim2);
            allreduceKernelWithLNEntryPoint<<<nblocks, nthreads, 0, stream>>>(
                smInputBuffChannelHandlesCuda, smOutputBuffChannelHandlesCuda, syncersCuda, globalSyncerCuda,
                nchannels, local_offset, rank, nranks, nelem_per_shard, input, output, output_before_ln.ptr, 
                ln_weight.ptr, pllm_tensor_input.dim1 / nranks, 
                pllm_tensor_input.dim2, epsilon);
        } else{
            allreduceKernelEntryPoint<<<nblocks, nthreads, 0, stream>>>(
                smInputBuffChannelHandlesCuda, smOutputBuffChannelHandlesCuda, syncersCuda, globalSyncerCuda,
                nchannels, local_offset, rank, nranks, nelem_per_shard, input, output);
        }
    }
};


class NetAsyncWrapper: public NetWrapper
{

    public:
    std::vector<Element*> remoteInputBuffs;
    std::vector<mscclpp::DeviceHandle<mscclpp::SimpleProxyChannel>> proxyChannelHandles;

    NetAsyncWrapper(){}
    void init(int rank, int nranks,
              Element* input, Element* output, int input_size, int output_size){
        NetWrapper::init(rank, nranks, input, output, input_size, output_size);
        remoteInputBuffs = getRemoteBuff(input);
        assert(remoteInputBuffs.size() == nranks - 1);
    }
    ~NetAsyncWrapper(){
        for (size_t i = 0; i < remoteInputBuffs.size(); ++i) {
#ifdef ENABLE_MPI
            CUDA_CHECK(cudaIpcCloseMemHandle(remoteInputBuffs[i]));
#endif
        }
    }
    virtual void start(cudaStream_t stream, int nblocks = 1, int nthreads = 8){}
    virtual void finish(cudaStream_t stream, int nblocks = 8, int nthreads = 1024){}

    protected:
    std::vector<Element*> getRemoteBuff(Element* localBuff) {
        std::vector<Element*> remoteBuffs;
        for (size_t r = 0; r < nranks; ++r) {
#ifdef ENABLE_MPI
            cudaIpcMemHandle_t handle;
#else
            static Element* globalBuff;
#endif
            if (r == rank) {
#ifdef ENABLE_MPI
                CUDA_CHECK(cudaIpcGetMemHandle(&handle, localBuff));
                MPI_Bcast(&handle, sizeof(cudaIpcMemHandle_t), MPI_BYTE, r, MPI_COMM_WORLD);
#else
                globalBuff = localBuff;
                worker_sync->barrier(); // wait for all peers to receive handle
                int device;
                CUDA_CHECK(cudaGetDevice(&device));
                std::cout << "rank " << rank << " setup handles on cuda dev " << device << std::endl;
                worker_sync->barrier(); // wait for all peers to done with the handle
#endif
            } else {
#ifdef ENABLE_MPI
                Element* remoteBuff;
                MPI_Bcast(&handle, sizeof(cudaIpcMemHandle_t), MPI_BYTE, r, MPI_COMM_WORLD);
                CUDA_CHECK(cudaIpcOpenMemHandle((void**) &remoteBuff, handle, cudaIpcMemLazyEnablePeerAccess));
#else
                worker_sync->barrier();
                Element* remoteBuff = globalBuff;
                worker_sync->barrier();
#endif
                remoteBuffs.push_back(remoteBuff);
            }
        }
        return remoteBuffs;
    }
    void setupProxyChannels(std::shared_ptr<mscclpp::ProxyService> service,
                            std::shared_ptr<mscclpp::Communicator> comm,
                            std::vector<std::shared_ptr<mscclpp::Connection>> connections,
                            mscclpp::DeviceHandle<mscclpp::SimpleProxyChannel>** proxyChannelHandlesCuda,
                            Element* input, Element* output, int input_size, int output_size) {
        const mscclpp::TransportFlags allTransports = mscclpp::Transport::CudaIpc;
        mscclpp::RegisteredMemory inputBuffRegMem = comm->registerMemory(input, input_size * sizeof(Element), allTransports);
        mscclpp::RegisteredMemory outputBuffRegMem;
        if (input != output) outputBuffRegMem = comm->registerMemory(output, output_size * sizeof(Element), allTransports);

        std::vector<mscclpp::NonblockingFuture<mscclpp::RegisteredMemory>> remoteRegMemories;
        mscclpp::RegisteredMemory& localRegMemory = (input != output) ? outputBuffRegMem : inputBuffRegMem;

        for (size_t r = 0; r < nranks; ++r) {
            if (r == rank) continue;
            comm->sendMemoryOnSetup(localRegMemory, r, 0);
            auto remoteMemory = comm->recvMemoryOnSetup(r, 0);
            remoteRegMemories.push_back(remoteMemory);
        }
        comm->setup();
        for (size_t i = 0; i < connections.size(); ++i) {
            proxyChannelHandles.push_back(mscclpp::deviceHandle(mscclpp::SimpleProxyChannel(
                service->proxyChannel(service->buildAndAddSemaphore(*comm, connections[i])),
                service->addMemory(remoteRegMemories[i].get()), service->addMemory(inputBuffRegMem)
            )));
        }
        comm->setup();

        assert(connections.size() == nranks - 1);
        CUDA_CHECK(cudaMalloc(proxyChannelHandlesCuda, (nranks - 1) * sizeof(mscclpp::DeviceHandle<mscclpp::SimpleProxyChannel>)));
        CUDA_CHECK(cudaMemcpy(*proxyChannelHandlesCuda, &proxyChannelHandles[proxyChannelHandles.size() - (nranks - 1)],
                              (nranks - 1) * sizeof(mscclpp::DeviceHandle<mscclpp::SimpleProxyChannel>), cudaMemcpyHostToDevice));
    }
};
