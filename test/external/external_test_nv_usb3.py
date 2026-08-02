import ctypes, unittest
import numpy as np

from tinygrad import Device, Tensor
from tinygrad.runtime.autogen import libusb, pci, nv_570 as nv_gpu
from tinygrad.runtime.ops_nv import USBIface
from tinygrad.runtime.support.usb import CustomASM24Controller, USB3

USB_IDS = {(0xADD1, 0x0001), (0x3801, 0x0001)}
RTX_3090_PCI_ID = 0x220410DE
CHRAM_FAULT_BITS = (1 << 4) | (1 << 5) | (1 << 12)


class TestNVUSB3(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    if not (USB3.list_devices(0xADD1, 0x0001) + USB3.list_devices(0x3801, 0x0001)):
      raise unittest.SkipTest("no ASM2464 USB device found")
    cls.dev = Device["NV"]
    if not isinstance(cls.dev.iface, USBIface): raise unittest.SkipTest("run with DEV=USB+NV:NAK")
    cls.iface = cls.dev.iface

  def test_device_and_firmware_identity(self):
    raw_usb, controller = self.iface.pci_dev.usb.usb, self.iface.pci_dev.usb
    usb_dev = libusb.libusb_get_device(raw_usb.handle)
    descriptor = libusb.struct_libusb_device_descriptor()
    self.assertEqual(libusb.libusb_get_device_descriptor(usb_dev, ctypes.byref(descriptor)), 0)
    self.assertIn((int(descriptor.idVendor), int(descriptor.idProduct)), USB_IDS)
    self.assertEqual(libusb.libusb_get_device_speed(usb_dev), libusb.LIBUSB_SPEED_SUPER)
    self.assertEqual(self.iface.pci_dev.read_config(pci.PCI_VENDOR_ID, 4), RTX_3090_PCI_ID)
    self.assertEqual(controller.firmware_protocol, (1, 0))
    self.assertGreaterEqual(controller.firmware_revision, 3)
    self.assertEqual(controller.firmware_capabilities & CustomASM24Controller.FW_REQUIRED_CAPABILITIES,
                     CustomASM24Controller.FW_REQUIRED_CAPABILITIES)

  def test_compute_and_exact_transfer(self):
    self.assertEqual((Tensor([1., 2., 3., 4.], device="NV") * 3 + 1).tolist(), [4., 7., 10., 13.])
    source = np.arange((4 << 20) // 4, dtype=np.uint32) ^ np.uint32(0xA5A55A5A)
    result = Tensor(source, device="NV").contiguous().realize().numpy()
    np.testing.assert_array_equal(result, source)

  def test_channel_and_aer_health(self):
    self.dev.synchronize()
    base_index, chram = 0, None
    while chram is None:
      table = self.iface.rm_control(self.dev.subdevice, nv_gpu.NV2080_CTRL_CMD_FIFO_GET_DEVICE_INFO_TABLE,
        nv_gpu.NV2080_CTRL_FIFO_GET_DEVICE_INFO_TABLE_PARAMS(baseIndex=base_index))
      entry = next((x for x in table.entries[:table.numEntries]
                    if x.engineData[3] == self.iface.runlist_id and x.engineData[11] == self.iface.runlist_pri_base), None)
      if entry is not None: chram = self.iface.pci_dev.map_bar(0, fmt='I', off=int(entry.engineData[14]), size=0x2000)
      elif not table.bMore or not table.numEntries: self.fail("graphics CHRAM table entry unavailable")
      else: base_index += table.numEntries
    for name in ("compute", "dma"):
      fifo = getattr(self.dev, f"{name}_gpfifo")
      state = chram[self.iface.gpfifo_submit_token(fifo.token) & 0x7ff]
      self.assertEqual(state & CHRAM_FAULT_BITS, 0, f"{name} CHRAM fault bits set: {state:#x}")
    for bus in range(self.iface.pci_dev.gpu_bus + 1):
      uncorrectable = self.iface.pci_dev.usb.pcie_cfg_req(0x104, bus=bus, size=4)
      correctable = self.iface.pci_dev.usb.pcie_cfg_req(0x110, bus=bus, size=4)
      self.assertEqual((uncorrectable, correctable), (0, 0), f"PCIe AER fault on bus {bus}")


if __name__ == "__main__": unittest.main()
