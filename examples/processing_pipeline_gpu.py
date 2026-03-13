"""
Example processing pipeline for phase data on the GPU:
- Load with Zarr in GPU
- Field retrieval
- Phase unwrapping
- Write to Zarr
"""

import zarr
import qpretrieve
import unwrap_phase_gpu as upg

# set cupy backend
upg.set_ndarray_backend("cupy")
qpretrieve.set_ndarray_backend("cupy")
xp = upg.get_ndarray_backend()

# load data with zarr via GPU memory
zarr.config.enable_gpu()

# load to cpu (mightn't need this)
edata = xp.load("./data/hologram_cell.npz")

# load to gpu
store_holo = zarr.storage.MemoryStore()
store_bg = zarr.storage.MemoryStore()
data_holo = zarr.create_array(store=store_holo, data=edata["data"])
data_bg = zarr.create_array(store=store_bg, data=edata["bg_data"])

# field retrieval on the GPU
holo = qpretrieve.OffAxisHologram(data_holo[:])
bg = qpretrieve.OffAxisHologram(data_bg[:])
holo.run_pipeline(filter_name="disk", filter_size=1 / 2, scale_to_filter=True)
bg.process_like(holo)
phase_wrapped = xp.asarray(holo.phase - bg.phase).astype(xp.float32)

# unwrap the phase on the GPU
phase_unwrapped = upg.algo_ls_poisson_periodic_grad(phase_wrapped)

# write from GPU to Zarr file
path = "unwrapped_cell_phase.zip"
with zarr.storage.ZipStore(path, mode='w') as disk:
    data_phase = zarr.create_array(store=disk, data=phase_unwrapped)

# reopen the file to ensure it is correct
with zarr.storage.ZipStore(path, mode='r') as disk_reload:
    data_reload = zarr.open_array(store=disk_reload, mode='r')
    assert data_phase.shape == data_reload.shape
    assert xp.allclose(phase_unwrapped, data_reload[:])
