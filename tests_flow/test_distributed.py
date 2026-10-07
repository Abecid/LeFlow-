import torch

from flow_jepa.common import distributed


def test_explicit_cuda_selects_device_before_any_probe(monkeypatch):
    monkeypatch.setenv("FLOW_DEVICE", "cuda")
    monkeypatch.setenv("WORLD_SIZE", "4")
    monkeypatch.setenv("RANK", "3")
    monkeypatch.setenv("LOCAL_RANK", "3")
    monkeypatch.delenv("FLOW_EGL_DEVICES", raising=False)
    calls = []

    def early_probe():
        raise AssertionError("CUDA probe before rank device selection")

    monkeypatch.setattr(torch.cuda, "is_available", early_probe)
    monkeypatch.setattr(torch.cuda, "device_count", early_probe)
    monkeypatch.setattr(torch.cuda, "set_device", lambda d: calls.append(("device", d)))
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: False)
    monkeypatch.setattr(
        torch.distributed, "init_process_group",
        lambda backend, timeout: calls.append(("backend", backend)),
    )
    assert distributed() == (3, 4, torch.device("cuda:3"))
    assert calls == [("device", torch.device("cuda:3")), ("backend", "nccl")]


def test_explicit_cpu_never_touches_cuda(monkeypatch):
    monkeypatch.setenv("FLOW_DEVICE", "cpu")
    monkeypatch.setenv("WORLD_SIZE", "1")
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.delenv("FLOW_EGL_DEVICES", raising=False)

    def reject():
        raise AssertionError("CPU test probed CUDA")

    monkeypatch.setattr(torch.cuda, "is_available", reject)
    assert distributed() == (0, 1, torch.device("cpu"))
