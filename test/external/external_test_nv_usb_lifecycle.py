import os, select, signal, subprocess, sys, time, unittest

from tinygrad.runtime.support.usb import USB3


CHILD_IMPORTS = """
import numpy as np

from tinygrad import Device, Tensor
from tinygrad.runtime.ops_nv import USBIface
from tinygrad.runtime.support.nv.nvdev import NVLifecycleState
"""

COMPUTE_WORKLOAD = """
assert (Tensor([1., 2., 3., 4.], device="NV") * 3 + 1).tolist() == [4., 7., 10., 13.]
source = np.arange((4 << 20) // 4, dtype=np.uint32) ^ np.uint32(0xA5A55A5A)
np.testing.assert_array_equal(Tensor(source, device="NV").contiguous().realize().numpy(), source)

dev = Device["NV"]
assert isinstance(dev.iface, USBIface)
dev.synchronize()
"""

COMPUTE_AND_CLOSE = CHILD_IMPORTS + COMPUTE_WORKLOAD + """
dev.finalize()
assert dev.iface.dev_impl.lifecycle_state is NVLifecycleState.CLOSED
"""

COMPUTE_AND_ATEXIT = """
import atexit

dev = None

def verify_atexit_teardown():
  assert dev is not None and dev._finalized, "tinygrad's atexit handler did not finalize the NVIDIA device"
  assert dev.iface.dev_impl.lifecycle_state is NVLifecycleState.CLOSED, "NVIDIA lifecycle did not close during atexit"
  print("NV_USB_ATEXIT_CLOSED", flush=True)

# Register before importing tinygrad so this verifier runs after tinygrad's LIFO atexit handler.
atexit.register(verify_atexit_teardown)
""" + CHILD_IMPORTS + COMPUTE_WORKLOAD

CRASH_AFTER_READY = CHILD_IMPORTS + COMPUTE_WORKLOAD + """
import time

from tinygrad.runtime.support.nv.ip import NVGspLifecycleState

assert dev.iface.dev_impl.flcn.attach_lifecycle_record() is NVGspLifecycleState.PROCESS_READY
assert dev.iface.dev_impl.NV_PFB_PRI_MMU_WPR2_ADDR_HI.read() != 0
print("NV_USB_PROCESS_READY", flush=True)
while True: time.sleep(60)
"""

CHECK_FINAL_STATE = """
import json

from tinygrad.runtime.support.nv.ip import NV_FLCN
from tinygrad.runtime.support.system import USBPCIDevice
from tinygrad.runtime.support.usb import USB3

devices = USB3.list_devices(0xADD1, 0x0001) + USB3.list_devices(0x3801, 0x0001)
assert len(devices) == 1, f"expected one ASM2464 USB device, found {len(devices)}"
pci_dev = USBPCIDevice("NV", *devices[0])
assert pci_dev.map_bar(0, off=0x1fa828, size=4, fmt="I")[0] == 0, "WPR2 remains active after process exit"
record, _ = pci_dev.gsp_lifecycle_view()
assert bytes(record[:len(NV_FLCN.LIFECYCLE_MAGIC)]) == bytes(len(NV_FLCN.LIFECYCLE_MAGIC)), "lifecycle record remains after process exit"
aer = [(pci_dev.usb.pcie_cfg_req(0x104, bus=bus, size=4), pci_dev.usb.pcie_cfg_req(0x110, bus=bus, size=4))
       for bus in range(pci_dev.gpu_bus + 1)]
assert all(status == (0, 0) for status in aer), f"PCIe AER fault after process exit: {aer}"
print("NV_USB_FINAL_STATE " + json.dumps({"wpr2_hi": 0, "record_magic": 0, "aer": aer}), flush=True)
"""

class TestNVUSBLifecycle(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    if not (USB3.list_devices(0xADD1, 0x0001) + USB3.list_devices(0x3801, 0x0001)):
      raise unittest.SkipTest("no ASM2464 USB device found")

  def run_child(self, script:str, label:str) -> str:
    env = {**os.environ, "DEV":"USB+NV:NAK"}
    proc = subprocess.run([sys.executable, "-c", script], env=env, capture_output=True, text=True, timeout=240)
    self.assertEqual(proc.returncode, 0, f"{label} failed\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}")
    self.assertNotIn("Exception ignored in atexit callback", proc.stderr,
                     f"{label} failed during atexit\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}")
    return proc.stdout

  def kill_child_after_marker(self, script:str, marker:str, label:str) -> str:
    env = {**os.environ, "DEV":"USB+NV:NAK"}
    proc = subprocess.Popen([sys.executable, "-c", script], env=env, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, bufsize=1)
    output:list[str] = []
    assert proc.stdout is not None
    try:
      deadline = time.monotonic() + 240
      while marker not in "".join(output):
        remaining = deadline - time.monotonic()
        if remaining <= 0 or not select.select([proc.stdout], [], [], max(0, remaining))[0]:
          self.fail(f"{label} timed out waiting for {marker}\noutput:\n{''.join(output)}")
        line = proc.stdout.readline()
        if not line:
          self.fail(f"{label} exited before {marker}\noutput:\n{''.join(output)}")
        output.append(line)
      proc.kill()
      tail, _ = proc.communicate(timeout=30)
      output.append(tail)
    finally:
      if proc.poll() is None:
        proc.kill()
        tail, _ = proc.communicate(timeout=30)
        output.append(tail)
    self.assertEqual(proc.returncode, -signal.SIGKILL, f"{label} was not killed as expected\noutput:\n{''.join(output)}")
    return "".join(output)

  def test_natural_atexit_and_explicit_reopen(self):
    self.assertIn("NV_USB_ATEXIT_CLOSED", self.run_child(COMPUTE_AND_ATEXIT, "natural atexit close"))
    self.run_child(COMPUTE_AND_CLOSE, "fresh-process explicit close")
    self.run_child(CHECK_FINAL_STATE, "final-state check")

  def test_sigkill_owner_recovers_in_next_process(self):
    self.kill_child_after_marker(CRASH_AFTER_READY, "NV_USB_PROCESS_READY", "abandoned owner")
    self.run_child(COMPUTE_AND_CLOSE, "post-crash recovery")
    self.run_child(CHECK_FINAL_STATE, "post-recovery final-state check")


if __name__ == "__main__": unittest.main()
