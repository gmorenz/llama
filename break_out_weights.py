import torch
import sys
import os
from math import prod


assert len(sys.argv) >= 3, "Usage: break_out_weights.py existing_weights_file.pth ... target_dir"

target_dir = sys.argv.pop()
sys.argv.pop(0)
checkpoints = sys.argv
os.makedirs(target_dir, exist_ok=True)
    
shortname_to_dim = {
    "w1": 0,
    "w2": 1,
    "w3": 0,
    "wo": 1,
    "wq": 0,
    "wk": 0,
    "wv": 0,
    "output": 0,
    "tok_embeddings": 1,
    "ffn_norm": None,
    "attention_norm": None,
    "norm": None,
    "rope": None,
}

def check_stride(t):
    for i in range(len(t.size())):
        assert t.stride()[i] == prod(t.size()[i+1:])

for i, checkpoint in enumerate(sorted(checkpoints)):
    print(f"Loading {checkpoint}")
    weights = torch.load(checkpoint, map_location="cpu")
    for name, param in weights.items():
        print(i, name, param.size())
        assert isinstance(name, str)
        assert isinstance(param, torch.Tensor)

        # We need to specify the out_storage *class* to be of the right kind, unfortunately
        # `v.storage()`'s class is the generic `TypedStorage` class and not the specific
        # `HalfStorage` class so we can't just reuse it's class to make a new one. Even though
        # the only thing `HalfStorage`  changes from `TypeStorage` appears to be the dtype.
        #
        # If we don't do this, `from_file` fails, because it has an assert that you're using
        # a subclass.
        #
        # Thus we just check we're only workign with half floats.
        if param.dtype != torch.half:
            # rope.freqs is in (at least some of) the weight files, 32 bit, and unused.
            # Just skip it and anything else like it.
            assert not name.startswith("layers.")
            assert name not in ['tok_embeddings.weight', 'norm.weight', 'output.weight']
            print(f"skipping {name}")
            continue 

        # Make sure param isn't sharing a storage
        if param.storage().size() != prod(param.size()):
            param = param.clone() 
        check_stride(param)

        out_path = os.path.join(target_dir, name)
        v_store = param.storage()

        # Figure out how to concatenate tensors from sharded
        # checkpoint files together.
        shortname = name.split('.')[-2]
        dim = shortname_to_dim[shortname]

        if dim is None and i != 0:
            # Non-sharded weights, and we've already saved them
            continue

        if dim is None or i == 0:
            # First time through, just save the weights to a file
            out_storage = torch.HalfStorage.from_file(out_path, shared=True, size=v_store.size())
            out_storage.copy_(v_store)            
            continue
        
        # We need to merge the loaded data with the data
        # we've already saved to disk, and save it back.

        # Calculate dim of, and load existing weights.
        size_multiple = lambda dim_i: 1 if dim_i != dim else i
        existing_size = tuple(x * size_multiple(i) for i, x in enumerate(param.size()))
        assert prod(existing_size) == v_store.size() * i
        existing_store = torch.HalfStorage.from_file(out_path, shared=False, size=prod(existing_size))
        existing_tensor = torch.HalfTensor()
        stride = torch.zeros(existing_size, device="meta").stride()
        existing_tensor.set_(existing_store, 0, existing_size, stride)
        check_stride(existing_tensor)

        # Concat with new weights
        new_tensor = torch.concat((existing_tensor, param), dim=dim)
        check_stride(new_tensor)
        
        # And save back to the file
        os.remove(out_path)
        out_storage = torch.HalfStorage.from_file(out_path, shared=True, size=new_tensor.storage().size())
        out_storage.copy_(new_tensor.storage())
    del weights