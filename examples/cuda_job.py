"""Run a small PyTorch operation on each CUDA card granted by reserve."""

from reserved.sdk import adopt_inherited_lease


if __name__ == "__main__":
    with adopt_inherited_lease() as reservation:
        import torch

        for grant, device in zip(reservation.devices, reservation.torch_devices()):
            reservation.check()
            total = torch.arange(4, device=device).sum().item()
            print(f"physical {grant.name} -> logical {device}: sum={total}")
