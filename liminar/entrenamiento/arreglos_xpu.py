"""Arreglos de FSDP2 para las GPU Intel (XPU) con oneCCL, los mismos de entrenar.py. Sin ellos, en Clementina hubo
fallas de página intermitentes en el primer paso de entrenamiento. Con GPU NVIDIA no hacen falta."""
import torch


def flujo_unico():
    """FSDP2 hace las comunicaciones en flujos propios (juntar pesos, reducir gradientes) para solaparlas con el
    cómputo. En las XPU, con oneCCL, eso dio fallas de página intermitentes en el primer paso de entrenamiento
    (escrituras en memoria ya liberada): todo va al flujo de cómputo, a costa del solapamiento."""
    from torch.distributed.fsdp._fully_shard import _fsdp_param_group as g
    original = g.FSDPCommContext.lazy_init

    def lazy_init(self, device):
        original(self, device)
        f = self.device_handle.current_stream()
        self.all_gather_copy_in_stream = self.all_gather_stream = f
        self.reduce_scatter_stream = self.all_reduce_stream = f
    g.FSDPCommContext.lazy_init = lazy_init


def colectivas_sincronas():
    """Cada colectiva de FSDP (juntar pesos, reducir gradientes) espera a terminar antes de seguir. oneCCL las ejecuta
    en su propia cola sin avisarle al asignador de torch, y FSDP libera y reusa los búferes apenas las encola."""
    from torch.distributed.fsdp._fully_shard import _fsdp_collectives as c
    for nombre in ('DefaultAllGather', 'DefaultReduceScatter', 'ProcessGroupAllocAllGather',
                   'ProcessGroupAllocReduceScatter'):
        clase = getattr(c, nombre, None)
        if clase is None:
            continue
        original = clase.__call__

        def envuelto(self, *args, _original=original, **kwargs):
            trabajo = _original(self, *args, **kwargs)
            if trabajo is not None:
                trabajo.wait()
            torch.xpu.synchronize()
            return trabajo
        clase.__call__ = envuelto


def aplicar():
    flujo_unico()
    colectivas_sincronas()
