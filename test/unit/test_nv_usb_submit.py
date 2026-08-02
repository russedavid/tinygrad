import types, unittest
from unittest.mock import MagicMock, patch

from tinygrad.runtime.ops_nv import GPFifo, NVCommandQueue, NVDevice, NVSignal, USBIface
from tinygrad.runtime.support.hcq import HCQSignal


class RecordingMMIO:
  def __init__(self, events, name): self.events, self.name, self.values = events, name, {}
  def __setitem__(self, index, value):
    self.values[index] = value
    self.events.append((self.name, index, value))
  def __getitem__(self, index):
    self.events.append((f"{self.name}_read", index))
    return self.values[index]


class TestNVUSBSubmission(unittest.TestCase):
  def test_gpfifo_diagnostics_allow_non_pci_allocation_metadata(self):
    dev = object.__new__(NVDevice)
    dev.nvdevice, dev.virtmem = 0xC1000001, 0xC1000002
    dev.iface = MagicMock(gpfifo_class=0xC36F, dma_class=0xC3B5)
    dev.iface.alloc.return_value.meta.hMemory = 0xC1000003
    dev.iface.rm_alloc.side_effect = [0xC1000004, 0xC1000005]
    dev.iface.rm_control.return_value.workSubmitToken = 0x1234
    area = MagicMock(va_addr=0x100000, meta=types.SimpleNamespace(hMemory=0xC1000006))

    fifo = dev._new_gpu_fifo(area, ctxshare=0, channel_group=0xC1000007, offset=0x2000, entries=16)

    self.assertIsNone(fifo.ring_paddr)
    self.assertIsNone(fifo.userd_paddr)

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
    runlist_mmio, chram_mmio = object(), object()
    iface.dev = types.SimpleNamespace(subdevice=0xC1000002)
    iface.dev_impl = types.SimpleNamespace(chip_name="GA102")
    iface.pci_dev = types.SimpleNamespace(map_bar=MagicMock(side_effect=[runlist_mmio, chram_mmio]))

    def rm_control(obj, cmd, params):
      self.assertEqual(obj, iface.dev.subdevice)
      if cmd == 0x20800179:
        self.assertEqual(params.engineList[0], 1)
        params.runlistPriBase[0], params.runlistId[0] = 0x2A0000, 3
      elif cmd == 0x20801112:
        params.numEntries = 1
        params.entries[0].engineData[3] = 3
        params.entries[0].engineData[11] = 0x2A0000
        params.entries[0].engineData[14] = 0x2B0000
      else: self.fail(f"unexpected RM control {cmd:#x}")
      return params
    iface.rm_control = rm_control

    usermode, mapped = iface.setup_usermode()

    self.assertEqual((usermode, mapped), (0xCE000000, runlist_mmio))
    self.assertEqual((iface.runlist_pri_base, iface.runlist_id, iface.chram_pri_base), (0x2A0000, 3, 0x2B0000))
    self.assertEqual(iface.pci_dev.map_bar.call_args_list, [
      unittest.mock.call(bar=0, fmt='I', off=0x2A0000, size=0x1000),
      unittest.mock.call(bar=0, fmt='I', off=0x2B0000, size=0x2000)])
    self.assertIs(iface.chram_mmio, chram_mmio)
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

  def test_signal_timeout_includes_usb_gpfifo_diagnostics(self):
    signal = object.__new__(NVSignal)
    signal.should_return = False
    signal.owner = types.SimpleNamespace(iface=types.SimpleNamespace(gpfifo_diagnostics=lambda: "USB GPFIFO state: GPGet=0"))

    with patch.object(HCQSignal, "wait", side_effect=RuntimeError("Wait timeout")):
      with self.assertRaisesRegex(RuntimeError, "Wait timeout; USB GPFIFO state: GPGet=0"):
        signal.wait(2)

  def test_gpfifo_diagnostics_include_rm_queue_identity(self):
    iface = object.__new__(USBIface)
    events = []
    ring, gpput, gpget = (RecordingMMIO(events, name) for name in ("ring", "gpput", "gpget"))
    ring.values[0], gpput.values[0], gpget.values[0] = 0x123, 1, 0
    fifo = GPFifo(ring, gpput, 16, token=0x2000034, gpget=gpget, put_value=1, channel=0xC1000042,
                  ring_paddr=0x210000, userd_paddr=0x210080)
    iface.dev = types.SimpleNamespace(compute_gpfifo=fifo)
    iface._runlist_doorbell = True
    iface.chram_mmio = RecordingMMIO(events, "chram")
    iface.chram_mmio.values[0x34] = 0x1152
    iface.pci_dev = types.SimpleNamespace(gpu_bus=0, usb=types.SimpleNamespace(pcie_cfg_req=lambda *args, **kwargs: 0))

    self.assertEqual(iface.gpfifo_health(), ({"compute": 0x1152}, [(0, 0, 0)]))
    detail = iface.gpfifo_diagnostics()

    self.assertIn("channel=0xc1000042, token=0x2000034, doorbell_token=0x34", detail)
    self.assertIn("ring_paddr=0x210000, userd_paddr=0x210080", detail)
    self.assertIn("host_put=0x1, GPPut=0x1, GPGet=0x0", detail)
    self.assertIn("CHRAM=0x1152[enable=1, next=0, busy=0, pbdma_faulted=1, eng_faulted=0, on_pbdma=1, on_eng=0, pending=1, acquire_fail=1]", detail)


if __name__ == "__main__": unittest.main()
