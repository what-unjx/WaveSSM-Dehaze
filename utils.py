import time
import numpy as np
import torch.nn as nn

# Key prefixes introduced by wrappers around the raw ConvIR module:
#   ``module.``    - DistributedDataParallel
#   ``_orig_mod.`` - torch.compile (OptimizedModule)
_WRAPPER_PREFIXES = ('module.', '_orig_mod.')


def _strip_wrapper_prefixes(key: str) -> str:
    """Strip all leading DDP/compile wrapper prefixes from a state-dict key."""
    stripped = True
    while stripped:
        stripped = False
        for prefix in _WRAPPER_PREFIXES:
            if key.startswith(prefix):
                key = key[len(prefix):]
                stripped = True
    return key


def normalize_state_dict(state_dict: dict) -> dict:
    """Return a checkpoint dict with bare module names as keys.

    Removes the ``module.`` (DDP) and ``_orig_mod.`` (torch.compile) prefixes so
    checkpoints stay byte-compatible with the un-wrapped model, regardless of
    how the live model object was wrapped.

    Args:
        state_dict: Raw ``model.state_dict()`` of any wrapped/unwrapped module.

    Returns:
        A new dict whose keys are the bare parameter/buffer names.
    """
    return {_strip_wrapper_prefixes(k): v for k, v in state_dict.items()}


def load_model_state(module: nn.Module, state_dict: dict) -> None:
    """Load a checkpoint into ``module`` regardless of DDP/torch.compile wrapping.

    Accepts checkpoints saved with or without wrapper prefixes and adapts them
    to whatever prefix layout ``module`` itself expects (strict load).

    Args:
        module: Target module (raw, OptimizedModule, DDP or DDP(OptimizedModule)).
        state_dict: Raw checkpoint weights (any prefix layout).

    Returns:
        None.
    """
    bare = normalize_state_dict(state_dict)
    try:
        module.load_state_dict(bare)
        return
    except RuntimeError:
        pass

    # The module is wrapped (its own keys carry prefixes): re-attach its exact prefixes.
    lookup = bare
    remapped = {}
    own_keys = list(module.state_dict().keys())
    for key in own_keys:
        source = _strip_wrapper_prefixes(key)
        if source not in lookup:
            raise KeyError(
                'checkpoint is missing key %r (expected under %r); '
                'not a %s checkpoint' % (source, key, type(module).__name__)
            )
        remapped[key] = lookup[source]
    module.load_state_dict(remapped)


class Adder(object):
    def __init__(self):
        self.count = 0
        self.num = float(0)




    def reset(self):
        self.count = 0
        self.num = float(0)




    def __call__(self, num: int):
        self.count += 1
        self.num += num




    def average(self) -> float:
        return self.num / self.count






class Timer(object):
    def __init__(self, option: str = 's'):
        self.tm = 0
        self.option = option
        if option == 's':
            self.devider = 1
        elif option == 'm':
            self.devider = 60
        else:
            self.devider = 3600




    def tic(self):
        self.tm = time.time()




    def toc(self) -> float:
        return (time.time() - self.tm) / self.devider






def check_lr(optimizer) -> float:
    lr = 0.0
    for i, param_group in enumerate(optimizer.param_groups):
        lr = param_group['lr']
    return lr
