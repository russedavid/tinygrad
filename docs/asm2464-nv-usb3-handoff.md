# ASM2464 + RTX 3090 USB3 technical handoff

This document describes the current tinygrad compatibility work for an NVIDIA RTX 3090 connected through an ASM2464 USB-to-PCIe bridge. It is intended to let another agent continue on Linux without reconstructing the hardware findings.

## Repository and branch

- Fork: `https://github.com/russedavid/tinygrad.git`
- Backup branch: `backup/nv-usb-current`
- The existing `bounty/nv-usb3-wip` PR branch was not updated.
- No firmware changes are required for the currently working NVIDIA path.

Clone the backup directly:

```bash
git clone --branch backup/nv-usb-current https://github.com/russedavid/tinygrad.git
cd tinygrad
```

The separate firmware repository used during bring-up was `/Users/davidrussell/Documents/asm2464pd-firmware`, branch `bounty/nv-usb3-wip`, commit `4e856ec`. The failed F6 experiments are not part of the working solution.

## Current working architecture

### GSP RPC

GSP RPC is NVIDIA's control plane. Tinygrad uses it for firmware initialization, registry/system information, RM object allocation, address spaces, page directories, channels, context promotion, control calls, asynchronous errors/events, and teardown. Tensor payload bytes do **not** travel through GSP RPC.

The host-to-GSP command ring has distinct SRAM pages. The GSP-to-host status ring has four logical record slots folded over two physical controller-visible pages:

```text
logical slot:   0      1      2      3
physical page: A000   F000   A000   F000
```

Tinygrad intentionally keeps one additional logical credit occupied, limiting the GSP to two live response elements. Any contiguous one- or two-element record maps injectively to A000/F000, including ring wrap. The reader copies and validates all currently published records before returning credit.

The implementation also:

- Sends the registry table, avoiding the previous ~841-record NOCAT flood.
- Tracks the direct response cursor independently.
- Checks record signature, length, checksum, and physical capacity.
- Retries bounded torn-record reads.
- Releases the final reserved credit during GSP unload.
- Uses a 3-second general RPC timeout rather than the old 120-second development timeout.

Observed stress behavior was 303 response records across initialization and 256 repeated RPC wraps: 302 one-page records and one two-page `GSP_RUN_CPU_SEQUENCER` event. No response larger than two pages was observed. A three-page response remains unsupported and may block at the producer before the host can diagnose it.

### F2 tensor transfers

Normal tensor data uses F2 and the NVIDIA copy engine, not GSP RPC.

The GSP page table is at `0x213000`. GSP command pages were moved to `0x27b000` through `0x27f000`, freeing SRAM slots 5 through 29 as a contiguous 400 KiB transfer arena:

```text
slot 0       folded GSP response page
slot 4       GSP queue page table
slots 5-29   400 KiB F2 tensor-transfer arena
slots 30-31  GSP command header and record pages
```

Host to GPU:

```text
host RAM
  -> USB bulk OUT / F2
  -> ASM SRAM slots 5-29
  -> NVIDIA copy engine
  -> destination VRAM
```

GPU to host:

```text
source VRAM
  -> NVIDIA copy engine
  -> ASM SRAM slots 5-29
  -> ordered zero-dword completion at GPU-visible 0x82800c
  -> F2 releases USB bulk IN
  -> host RAM
```

The completion address overlaps four bytes of the directly visible GSP status header. Tinygrad saves those bytes before each transfer and restores them immediately afterward. Command binding is completed before F2 is armed, so later host bulk writes cannot replace the pending read.

Completion-coupled reads currently work only when F2 is armed from slot 0. Since the payload begins at slot 5, each read contains an 80 KiB prefix which the host discards. F2 is one-shot, so each 400 KiB chunk has its own copy, completion, and USB read cycle.

Set `NV_USB_F2=0` to retain the slower BAR1 staging fallback.

Measured 16 MiB transfers:

| Path | Host to GPU | GPU to host |
| --- | ---: | ---: |
| BAR1 staging | ~3.7 MB/s | ~1.8 MB/s |
| F2, 192 KiB arena | ~102 MB/s | ~6.3 MB/s |
| F2, 400 KiB arena | ~138 MB/s | ~12.8 MB/s |

## Linux compiler and renderer setup

On the original Apple-silicon macOS host, CUDA Toolkit components were unavailable. `DEV=USB+NV:NAK` used Mesa NIR/NAK and reached only about 4.91 TFLOPS on the completed 4096-square BEAM=3 FP16 GEMM. The NAK renderer currently advertises no tensor-core descriptions to tinygrad, so the optimizer cannot emit `Ops.WMMA` Tensor Core operations.

On Linux with CUDA Toolkit compiler components installed, do not force NAK initially. Tinygrad tries NVIDIA renderers in this order:

1. `CUDARenderer` using NVRTC.
2. `PTXRenderer` using nvJitLink.
3. `NVCCRenderer` using `nvcc`.
4. `NAKRenderer` as the toolkit-independent fallback.

Check the selected renderer:

```bash
DEV=USB+NV PYTHONPATH=. python - <<'PY'
from tinygrad import Device
dev = Device["NV"]
print(type(dev.renderer).__name__)
dev.finalize()
PY
```

Force a renderer when diagnosing:

```bash
DEV=USB+NV:CUDA  # NVRTC
DEV=USB+NV:PTX   # PTX + nvJitLink
DEV=USB+NV:NVCC  # nvcc
DEV=USB+NV:NAK   # Mesa NAK
```

Run BEAM=3 GEMMs without an external test timeout:

```bash
DEV=USB+NV:CUDA DEBUG=1 PYTHONPATH=. \
  python -m pytest \
  test/external/speed_v_theoretical.py::TestKernelSpeed::test_gemm_4096 \
  test/external/speed_v_theoretical.py::TestKernelSpeed::test_gemm_8192 \
  -s -q
```

On macOS/NAK, 4096 completed at 4.91 TFLOPS. The 8192 BEAM search reproducibly aborted inside repeated USB/libusb timeline polling after several minutes, both with default parallel compilation and with `PARALLEL=0`. No overall benchmark timeout was used.

## Reset and recovery commands

Run commands from the tinygrad repository with the desired Python environment active. Add `sudo -E` if the Linux USB permissions require it.

### Clean GSP teardown

Use this when the device is healthy:

```bash
DEV=USB+NV:NAK PYTHONPATH=. python - <<'PY'
from tinygrad import Device
dev = Device["NV"]
dev.finalize()
print("clean NVIDIA teardown complete")
PY
```

Healthy teardown sends the GSP unload RPC, performs FWSEC/booter unload, and verifies WPR2 is zero.

### Full bridge transfer reset + GPU power cycle + FLR + WPR verification

Use this after a failed transfer, abandoned process, stale WPR, or failed GSP setup:

```bash
PYTHONPATH=. python - <<'PY'
from tinygrad.runtime.autogen import libusb
from tinygrad.runtime.support.system import USBPCIDevice
from tinygrad.runtime.support.usb import CustomASM24Controller, USB3, checked

devices = USB3.list_devices(0xADD1, 0x0001) + USB3.list_devices(0x3801, 0x0001)
assert len(devices) == 1, f"expected one ASM2464, found {len(devices)}"

# Clear sticky F2 and bulk-endpoint state first.
controller = CustomASM24Controller(USB3(devices[0][0]))
controller.write(0xC412, b"\x00")
controller.write(0xC42A, b"\x01")
controller.write(0xC42A, b"\x00")
controller.write(0x9094, b"\x01")
controller.write(0x9094, b"\x08")
controller.write(0x9093, b"\x02")
checked(libusb.libusb_clear_halt)(controller.usb.handle, 0x81)
checked(libusb.libusb_clear_halt)(controller.usb.handle, 0x02)

# Reset the downstream GPU without physically power-cycling the system.
controller.set_pcie_power(False, timeout=1000)
controller.set_pcie_power(True, timeout=5000)
checked(libusb.libusb_release_interface)(controller.usb.handle, 0)
libusb.libusb_close(controller.usb.handle)

pci_dev = USBPCIDevice("NV", *devices[0])
if pci_dev.supports_flr():
  pci_dev.function_level_reset(timeout=2.0)

wpr2_hi = pci_dev.map_bar(0, off=0x1FA828, size=4, fmt="I")[0]
assert wpr2_hi == 0, f"WPR2 remains active: {wpr2_hi:#x}"

# Clear sticky AER status left by deliberately invalid PCIe probes.
for bus in range(pci_dev.gpu_bus + 1):
  for offset in (0x104, 0x110):
    status = pci_dev.usb.pcie_cfg_req(offset, bus=bus, size=4)
    if status:
      pci_dev.usb.pcie_cfg_req(offset, bus=bus, value=status, size=4)

checked(libusb.libusb_release_interface)(pci_dev.usb.usb.handle, 0)
libusb.libusb_close(pci_dev.usb.usb.handle)
print("bridge/GPU recovery complete; WPR2 is clear")
PY
```

### ASM controller CPU restart

Use this only when the transfer engine remains sticky after the full recovery above, or after a process aborts inside libusb. `CC31` restarts the ASM controller CPU and causes USB re-enumeration. It does **not** itself clear GPU WPR; run the full recovery command afterward.

Run the controller restart in a separate process because its USB handle becomes invalid:

```bash
PYTHONPATH=. python - <<'PY'
import time
from tinygrad.runtime.support.usb import CustomASM24Controller, USB3

devices = USB3.list_devices(0xADD1, 0x0001) + USB3.list_devices(0x3801, 0x0001)
assert len(devices) == 1, f"expected one ASM2464, found {len(devices)}"
controller = CustomASM24Controller(USB3(devices[0][0]))
controller.write(0xCC31, b"\x01")
time.sleep(2.0)
print("ASM controller CPU reset requested; USB should be re-enumerated")
PY
```

Then run the full bridge/GPU recovery command from the previous section.

## Validation commands

Focused unit tests, with the repository-required worker count:

```bash
python -m pytest \
  test/unit/test_nv_gsp_boot.py \
  test/unit/test_nv_rpc.py \
  test/unit/test_nv_usb_submit.py \
  test/unit/test_usb.py \
  -x -q -n12
```

Hardware device, transfer, channel, and AER checks:

```bash
DEV=USB+NV:NAK PYTHONPATH=. \
  python -m pytest test/external/external_test_nv_usb3.py -x -q
```

Lifecycle, atexit, SIGKILL recovery, and setup-failure recovery:

```bash
PYTHONPATH=. python -m pytest \
  test/external/external_test_nv_usb_lifecycle.py -x -q
```

16 MiB bidirectional transfer benchmark:

```bash
DEV=USB+NV:NAK NSZ=2048 PYTHONPATH=. \
  python -m pytest \
  test/speed/external_test_copy_speed.py::TestCopySpeed::testCopyCPUtoDefault \
  test/speed/external_test_copy_speed.py::TestCopySpeed::testCopyDefaulttoCPU \
  -s -q
```

Static checks:

```bash
python -m mypy tinygrad/
python -m ruff check .
git diff --check
```

## Known negative results

Do not repeat these unchanged:

- The historical firmware F6 handler never read arbitrary SRAM on hardware; its scratch data was stale USB-output data.
- Remapping B26F from `0x28` to `0x22` removes the directly visible GSP status-header aperture and stalls RPC command consumption.
- GSP status `writePtr` is at `+0x10`; the actual command `readPtr` is in the receive header at `+0x20`, not `+0x0c`.
- Legacy MSI cannot synthesize the NVMe completion transaction required by F2.
- A true two-entry GSP status ring has only one usable credit and stalls on the two-page boot CPU-sequencer event.
- B000/B100 admin queue windows were not GPU-visible in the current bridge mode.
- Switching RTX 3090 BAR1 to system-memory target caused nested reads to hang and destabilized teardown.
- F2 reads armed directly at slot 5 did not release with the current firmware. Slot-0-prefixed reads do work.
- A 32-slot F2 read stalled; a 30-slot read was verified.

## Promising next work

1. Run CUDA/NVRTC or PTX/nvJitLink renderers on Linux and measure Tensor Core GEMM throughput over the same USB transport.
2. Add a backward-compatible firmware request for selected-slot bulk IN. The current F2 handler writes `C429` after starting `C412`; program all source/command fields before starting the DMA engine. Prefer a new advertised vendor request rather than changing existing F2 semantics initially.
3. If selected-slot reads work, remove the 80 KiB download prefix.
4. Reduce per-chunk download submission overhead, possibly with pre-bound copy commands or firmware-assisted chained reads.
5. Keep the folded two-page GSP response transport scoped to the pinned firmware/current tinygrad RPC set unless a genuine additional visible page or arbitrary SRAM read mechanism is found.

## Current validated state

- GSP initialization is approximately six seconds.
- 400 KiB F2 upload/download arena is enabled by default.
- 16 MiB transfer benchmark: approximately 138 MB/s host-to-GPU and 12.8 MB/s GPU-to-host.
- Exact compute and 4 MiB round-trip passed.
- USB3 device/transfer/channel/AER suite passed.
- Lifecycle/crash recovery suite passed.
- Focused units, mypy, ruff, and `git diff --check` passed before this documentation-only commit.
