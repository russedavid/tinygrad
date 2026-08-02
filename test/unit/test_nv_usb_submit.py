import types, unittest
from unittest.mock import MagicMock, patch

from tinygrad.runtime.ops_nv import GPFifo, NVCommandQueue, USBIface


class RecordingMMIO:
  def __init__(self, events, name): self.events, self.name, self.values = events, name, {}
  def __setitem__(self, index, value):
    self.values[index] = value
    self.events.append((self.name, index, value))
  def __getitem__(self, index):
    self.events.append((f"{self.name}_read", index))
    return self.values[index]


class TestNVUSBSubmission(unittest.TestCase):
  def test_submission_without_pci_transport(self):
    events = []
    dev = types.SimpleNamespace(iface=types.SimpleNamespace(), gpu_mmio=RecordingMMIO(events, "doorbell"))
    queue = object.__new__(NVCommandQueue)
    queue.binded_device, queue.hw_page, queue._q = dev, types.SimpleNamespace(va_addr=0x1000), [0, 0]
    gpfifo = GPFifo(RecordingMMIO(events, "ring"), RecordingMMIO(events, "gpput"), 16, token=0x1234)

    with patch("tinygrad.runtime.ops_nv.System.memory_barrier", MagicMock()): queue._submit_to_gpfifo(dev, gpfifo)

    self.assertEqual([event[0] for event in events], ["ring", "gpput", "doorbell"])
    queue.binded_device = None

  def test_flushes_bar_writes_before_and_after_doorbell(self):
    events = []
    pci_dev = types.SimpleNamespace(flush_writes=lambda: events.append(("flush",)))
    dev = types.SimpleNamespace(iface=types.SimpleNamespace(pci_dev=pci_dev), gpu_mmio=RecordingMMIO(events, "doorbell"))
    queue = object.__new__(NVCommandQueue)
    queue.binded_device, queue.hw_page, queue._q = dev, types.SimpleNamespace(va_addr=0x1000), [0, 0, 0, 0]
    gpfifo = GPFifo(RecordingMMIO(events, "ring"), RecordingMMIO(events, "gpput"), 16, token=0x1234)

    with patch("tinygrad.runtime.ops_nv.System.memory_barrier", MagicMock()): queue._submit_to_gpfifo(dev, gpfifo)

    self.assertEqual([event[0] for event in events], ["ring", "gpput", "flush", "doorbell", "flush"])
    self.assertEqual(events[-2], ("doorbell", 0x90 // 4, 0x1234))
    queue.binded_device = None

  def test_submission_uses_transport_doorbell_token(self):
    events = []
    pci_dev = types.SimpleNamespace(flush_writes=lambda: events.append(("flush",)))
    iface = types.SimpleNamespace(pci_dev=pci_dev, gpfifo_submit_token=lambda token: token & 0x7ff)
    dev = types.SimpleNamespace(iface=iface, gpu_mmio=RecordingMMIO(events, "doorbell"))
    queue = object.__new__(NVCommandQueue)
    queue.binded_device, queue.hw_page, queue._q = dev, types.SimpleNamespace(va_addr=0x1000), [0, 0]
    gpfifo = GPFifo(RecordingMMIO(events, "ring"), RecordingMMIO(events, "gpput"), 16, token=0x2001234)

    with patch("tinygrad.runtime.ops_nv.System.memory_barrier", MagicMock()): queue._submit_to_gpfifo(dev, gpfifo)

    self.assertIn(("doorbell", 0x90 // 4, 0x234), events)
    queue.binded_device = None

  def test_verifies_bar_writes_before_doorbell(self):
    events = []
    pci_dev = types.SimpleNamespace(flush_writes=lambda: events.append(("flush",)), verify_bar_writes=True)
    dev = types.SimpleNamespace(iface=types.SimpleNamespace(pci_dev=pci_dev), gpu_mmio=RecordingMMIO(events, "doorbell"))
    queue = object.__new__(NVCommandQueue)
    queue.binded_device, queue.hw_page, queue._q = dev, types.SimpleNamespace(va_addr=0x1000), [0, 0]
    gpfifo = GPFifo(RecordingMMIO(events, "ring"), RecordingMMIO(events, "gpput"), 16, token=0x1234)

    with patch("tinygrad.runtime.ops_nv.System.memory_barrier", MagicMock()): queue._submit_to_gpfifo(dev, gpfifo)

    self.assertEqual([event[0] for event in events],
                     ["ring", "gpput", "flush", "ring_read", "gpput_read", "doorbell", "flush"])
    queue.binded_device = None

  def test_ampere_usb_uses_internal_runlist_doorbell(self):
    iface = object.__new__(USBIface)
    runlist_mmio = object()
    iface.dev = types.SimpleNamespace(subdevice=0xC1000002)
    iface.dev_impl = types.SimpleNamespace(chip_name="GA102")
    iface.pci_dev = types.SimpleNamespace(map_bar=MagicMock(return_value=runlist_mmio))

    def rm_control(obj, cmd, params):
      self.assertEqual(obj, iface.dev.subdevice)
      if cmd == 0x20800179:
        self.assertEqual(params.engineList[0], 1)
        params.runlistPriBase[0], params.runlistId[0] = 0x2A0000, 3
      else: self.fail(f"unexpected RM control {cmd:#x}")
      return params
    iface.rm_control = rm_control

    usermode, mapped = iface.setup_usermode()

    self.assertEqual((usermode, mapped), (0xCE000000, runlist_mmio))
    self.assertEqual((iface.runlist_pri_base, iface.runlist_id), (0x2A0000, 3))
    iface.pci_dev.map_bar.assert_called_once_with(bar=0, fmt='I', off=0x2A0000, size=0x1000)
    self.assertEqual(iface.gpfifo_submit_token(0x1205), 0x205)

  def test_blackwell_usb_keeps_virtual_function_doorbell(self):
    iface = object.__new__(USBIface)
    mmio = object()
    iface.dev_impl = types.SimpleNamespace(chip_name="GB202")
    iface.pci_dev = types.SimpleNamespace(map_bar=MagicMock(return_value=mmio))

    usermode, mapped = iface.setup_usermode()

    self.assertEqual((usermode, mapped), (0xCE000000, mmio))
    iface.pci_dev.map_bar.assert_called_once_with(bar=0, fmt='I', off=0xBB0000, size=0x10000)
    self.assertEqual(iface.gpfifo_submit_token(0x12345678), 0x12345678)

if __name__ == "__main__": unittest.main()
