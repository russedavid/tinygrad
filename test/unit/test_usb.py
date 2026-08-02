import unittest
from unittest.mock import MagicMock, patch

from tinygrad.runtime.autogen import pci
from tinygrad.runtime.ops_nv import NVAllocator, USBIface
from tinygrad.runtime.support.system import System, PCIIfaceBase, USBPCIDevice
from tinygrad.runtime.support.usb import ASM24GSPQueueInterface, CustomASM24Controller, USBMMIOInterface


class TestCustomASM24Controller(unittest.TestCase):
  @staticmethod
  def firmware_info(*, magic=b"TG24", major=1, minor=0, size=16, capabilities=0x7F, revision=3):
    return CustomASM24Controller.FW_INFO_STRUCT.pack(magic, major, minor, size, capabilities, revision)

  def test_waits_for_pcie_link_after_power_on(self):
    usb = MagicMock()
    usb.control_read.return_value = self.firmware_info()
    with patch.object(CustomASM24Controller, "read", MagicMock(side_effect=[b"\x59", b"\x59", b"\x78"])) as read, \
         patch.object(CustomASM24Controller, "set_pcie_power") as set_pcie_power, \
         patch("tinygrad.runtime.support.usb.time.sleep") as sleep:
      controller = CustomASM24Controller(usb, minimum_revision=3)

    self.assertEqual(controller.firmware_protocol, (1, 0))
    self.assertEqual(controller.firmware_capabilities, 0x7F)
    self.assertEqual(controller.firmware_revision, 3)
    usb.control_read.assert_called_once_with(0xF4, 16)
    set_pcie_power.assert_called_once_with(True)
    self.assertEqual(read.call_count, 3)
    sleep.assert_called_once_with(0.01)

  def test_handshake_is_optional_for_existing_amd_firmware(self):
    usb = MagicMock()
    with patch.object(CustomASM24Controller, "read", return_value=b"\x78"):
      controller = CustomASM24Controller(usb)

    usb.control_read.assert_not_called()
    self.assertIsNone(controller.firmware_protocol)
    self.assertIsNone(controller.firmware_capabilities)
    self.assertIsNone(controller.firmware_revision)

  def test_rejects_missing_firmware_handshake(self):
    usb = MagicMock()
    usb.control_read.side_effect = RuntimeError("pipe stalled")

    with self.assertRaisesRegex(RuntimeError, "request 0xF4 failed; flash revision 3 or newer"):
      CustomASM24Controller(usb, minimum_revision=3)

  def test_rejects_incompatible_firmware_info(self):
    cases = [
      (self.firmware_info(magic=b"BAD!"), "firmware information"),
      (self.firmware_info(major=2), "protocol 2.0"),
      (self.firmware_info(capabilities=0x3F), "missing required capabilities 0x40"),
      (self.firmware_info(revision=2), "revision 2 is too old"),
    ]
    for payload, message in cases:
      with self.subTest(message=message):
        usb = MagicMock()
        usb.control_read.return_value = payload
        with self.assertRaisesRegex(RuntimeError, message): CustomASM24Controller(usb, minimum_revision=3)

  def test_memory_tlp_format_tracks_address_width(self):
    controller = object.__new__(CustomASM24Controller)
    controller.usb = MagicMock()
    with patch.object(controller, "pcie_request", return_value=0):
      controller.pcie_mem_read(0x10000000, 4)
      controller.pcie_request.assert_called_once_with(0x00, 0x10000000)
      controller.pcie_request.reset_mock()
      controller.pcie_mem_read(0x800000000, 4)
      controller.pcie_request.assert_called_once_with(0x20, 0x800000000)

      controller.pcie_request.reset_mock()
      controller.pcie_mem_write(0x10000000, bytes(4))
      controller.pcie_request.assert_called_once_with(0x40, 0x10000000, 0)
      controller.pcie_request.reset_mock()
      controller.pcie_mem_write(0x800000000, bytes(4))
      controller.pcie_request.assert_called_once_with(0x60, 0x800000000, 0)

    controller.usb.bulk_read.return_value = memoryview(bytes(8))
    controller.pcie_mem_read(0x10000000, 8)
    self.assertEqual(controller.usb.control_write.call_args.args[:3], (0xF0, 0x0F00, 2))
    controller.pcie_mem_write(0x800000000, bytes(8))
    self.assertEqual(controller.usb.control_write.call_args.args[:3], (0xF0, 0x0F60, 1))
    controller.usb.bulk_write.assert_called_once_with(bytes(8), timeout=30000)

  def test_mmio_slice_preserves_element_format(self):
    controller = MagicMock()
    controller.pcie_mem_read.return_value = memoryview(bytes.fromhex("0100000002000000"))

    mmio = USBMMIOInterface(controller, 0x10000000, 8, fmt='I')
    self.assertEqual(mmio[:2], [1, 2])

  def test_unaligned_mmio_uses_aligned_read_modify_write(self):
    controller = MagicMock()
    controller.pcie_mem_read.return_value = memoryview(bytes.fromhex("0011223344556677"))
    mmio = USBMMIOInterface(controller, 0x10000001, 5, fmt='B')

    self.assertEqual(bytes(mmio[:5]), bytes.fromhex("1122334455"))
    controller.pcie_mem_read.assert_called_with(0x10000000, 8)

    mmio[1:4] = b"abc"
    controller.pcie_mem_write.assert_called_once_with(0x10000000, bytes.fromhex("0011616263556677"))

  def test_mmio_write_cannot_cross_the_mapped_window(self):
    controller = MagicMock()
    mmio = USBMMIOInterface(controller, 0x800000000, 0x100, fmt='B')

    with self.assertRaisesRegex(AssertionError, "USB MMIO write size mismatch"):
      mmio[0x100:0x104] = bytes(4)

    controller.pcie_mem_write.assert_not_called()

  def test_scsi_write_arm_encodes_slot_range_and_reuses_setup(self):
    controller = object.__new__(CustomASM24Controller)
    controller.usb = MagicMock()

    controller.scsi_write_arm(0x54000, start_slot=11)
    controller.usb.control_write.assert_called_once_with(0xF2, value=0x2A0, index=0x150B)

    controller.usb.bulk_write(bytes(0x54000))
    controller.usb.bulk_write(bytes(0x54000))
    self.assertEqual(controller.usb.control_write.call_count, 1)
    self.assertEqual(controller.usb.bulk_write.call_count, 2)

  def test_large_pcie_transfers_are_chunked(self):
    controller = object.__new__(CustomASM24Controller)
    controller.usb = MagicMock()
    size = controller.PCIE_BULK_CHUNK_SIZE + 8

    with patch.object(controller, "_f0_out") as f0_out:
      controller.pcie_mem_write(0x800000000, bytes(size))
      self.assertEqual([(c.args[0], c.args[2], c.args[3]) for c in f0_out.call_args_list],
                       [(0x60, 0x800000000, 0x40000), (0x60, 0x800100000, 2)])
      self.assertEqual([len(c.args[0]) for c in controller.usb.bulk_write.call_args_list], [1 << 20, 8])

      f0_out.reset_mock()
      controller.usb.bulk_read.side_effect = lambda nbytes, timeout: memoryview(bytes(nbytes))
      self.assertEqual(len(controller.pcie_mem_read(0x10000000, size)), size)
      self.assertEqual([(c.args[0], c.args[2], c.args[3]) for c in f0_out.call_args_list],
                       [(0x00, 0x10000000, 0x40000), (0x00, 0x10100000, 2)])


class FakePCIeController:
  def __init__(self): self.writes = []

  def pcie_cfg_req(self, byte_addr, bus=1, dev=0, fn=0, value=None, size=4):
    if value is not None:
      self.writes.append((byte_addr, bus, value, size))
      return None
    if byte_addr == pci.PCI_VENDOR_ID and size == 4:
      return {0: 0x24631B21, 1: 0x1B631B21, 2: 0x220410DE}[bus]
    if byte_addr == pci.PCI_HEADER_TYPE and size == 1:
      return pci.PCI_HEADER_TYPE_BRIDGE if bus < 2 else pci.PCI_HEADER_TYPE_NORMAL
    raise AssertionError(f"unexpected config read: {byte_addr=:#x}, {bus=}, {size=}")


class TestUSBPCIeDiscovery(unittest.TestCase):
  def test_nvidia_discovers_legacy_and_current_firmware_vids(self):
    legacy, current = (object(), "usb:legacy"), (object(), "usb:current")
    with patch("tinygrad.runtime.ops_nv.USB3.list_devices", side_effect=[[legacy], [current]]) as list_devices, \
         patch("tinygrad.runtime.ops_nv.hcq_filter_visible_devices", side_effect=lambda devices, _: devices), \
         patch("tinygrad.runtime.ops_nv.USBPCIDevice") as pci_device, patch("tinygrad.runtime.ops_nv.NVDev"), \
         patch.object(USBIface, "_init_nvd"):
      iface = USBIface(MagicMock(), 1)

    self.assertEqual(list_devices.call_args_list, [unittest.mock.call(0xADD1, 0x0001), unittest.mock.call(0x3801, 0x0001)])
    pci_device.assert_called_once_with("NV", *current)
    self.assertEqual(iface.count, 2)

  @patch.object(USBPCIDevice, "_setup_pcie")
  @patch("tinygrad.runtime.support.system.CustomASM24Controller")
  @patch("tinygrad.runtime.support.system.USB3")
  @patch.object(System, "flock_acquire", return_value=1)
  def test_nvidia_device_allows_full_fixed_window_boot_drain(self, flock_acquire, usb3, controller, setup_pcie):
    dev = USBPCIDevice("NV", MagicMock(), "custom v0.1")

    self.assertEqual(dev.gsp_rpc_timeout_ms, 120000)
    controller.assert_called_once_with(usb3.return_value, minimum_revision=3)

  def test_stops_before_writing_endpoint_bus_registers(self):
    controller = FakePCIeController()

    self.assertEqual(System.pci_find_usb_endpoint(controller), 2)
    self.assertEqual(controller.writes, [
      (pci.PCI_PRIMARY_BUS, 0, 0x00FF0100, 4),
      (pci.PCI_PRIMARY_BUS, 1, 0x00FF0201, 4),
    ])

  def test_bar_setup_programs_each_bridge_primary_bus(self):
    controller = MagicMock()
    controller.pcie_cfg_req.side_effect = lambda byte_addr, **kw: \
      (0 if byte_addr == 0x100 else pci.PCI_BASE_ADDRESS_SPACE_IO) if "value" not in kw else None

    System.pci_setup_usb_bars(controller, gpu_bus=2, mem_base=0x10000000, pref_mem_base=0x800000000)

    bus_writes = [c for c in controller.pcie_cfg_req.call_args_list if c.args[0] == pci.PCI_PRIMARY_BUS and "value" in c.kwargs]
    self.assertEqual([(c.kwargs["bus"], c.kwargs["value"]) for c in bus_writes], [(0, 0x00020100), (1, 0x00020201)])

  def test_stages_nvidia_gsp_arguments_in_fixed_xdata_window(self):
    dev = object.__new__(USBPCIDevice)
    dev.usb = MagicMock()

    self.assertEqual(dev.stage_gsp_rm_args(b"RM"), 0x828100)
    self.assertEqual(dev.stage_gsp_libos_args(b"OS"), 0x828200)

    self.assertEqual(dev.usb.write.call_args_list[0].args, (0xB900, b"RM" + bytes(0xFE)))
    self.assertEqual(dev.usb.write.call_args_list[1].args, (0xBA00, b"OS" + bytes(0xFE)))

  def test_map_bar_rejects_an_adjacent_bar_address(self):
    dev = object.__new__(USBPCIDevice)
    dev.usb, dev._bar_info = MagicMock(), {1: (0x800000000, 0x10000000)}

    with self.assertRaisesRegex(ValueError, "exceeds its 0x10000000-byte aperture"):
      dev.map_bar(1, off=0x10000000, size=4)

  @patch("tinygrad.runtime.support.system.time.sleep")
  def test_stages_gsp_boot_after_configuring_known_good_link_settings(self, sleep):
    dev = object.__new__(USBPCIDevice)
    dev.gpu_bus, dev.usb = 2, MagicMock()
    dev.usb.pcie_cfg_req.return_value = 0

    dev.stage_gsp_boot(b"boot")

    sleep.assert_called_once_with(0.1)
    writes = [(call.args[0], call.kwargs.get("bus"), call.kwargs.get("value"))
              for call in dev.usb.pcie_cfg_req.call_args_list if "value" in call.kwargs]
    self.assertIn((0x80 + 0x30, 1, 1), writes)
    self.assertIn((0x78 + 0x30, 2, 1), writes)
    self.assertIn((0x78 + 0x08, 2, 0), writes)
    self.assertIn((0x104, 2, 0xFFFFFFFF), writes)
    dev.usb.scsi_write.assert_called_once_with(b"boot")

  @patch("tinygrad.runtime.support.system.time.sleep")
  def test_reset_uses_upstream_bridge_without_power_cycling_usb(self, sleep):
    dev = object.__new__(USBPCIDevice)
    dev.gpu_bus, dev.usb = 2, MagicMock()
    dev.read_config, dev.write_config_flush, dev._setup_pcie = MagicMock(return_value=0x7), MagicMock(), MagicMock()
    dev.usb.pcie_cfg_req.return_value = pci.PCI_BRIDGE_CTL_VGA

    dev.reset()

    dev.write_config_flush.assert_called_once_with(pci.PCI_COMMAND, 0x3, 2)
    self.assertEqual(dev.usb.pcie_cfg_req.call_args_list, [
      unittest.mock.call(pci.PCI_BRIDGE_CONTROL, bus=1, size=2),
      unittest.mock.call(pci.PCI_BRIDGE_CONTROL, bus=1, value=pci.PCI_BRIDGE_CTL_VGA | pci.PCI_BRIDGE_CTL_BUS_RESET, size=2),
      unittest.mock.call(pci.PCI_BRIDGE_CONTROL, bus=1, value=pci.PCI_BRIDGE_CTL_VGA, size=2),
    ])
    dev.usb.wait_for_pcie_link.assert_called_once_with()
    self.assertEqual(sleep.call_args_list, [unittest.mock.call(0.1), unittest.mock.call(1.0)])
    dev._setup_pcie.assert_called_once_with()


class TestUSBIfaceAllocation(unittest.TestCase):
  def test_usb_copy_staging_pool_is_bounded_for_small_bar1(self):
    dev = MagicMock()
    dev.iface = object.__new__(USBIface)

    with patch("tinygrad.runtime.ops_nv.HCQAllocator.__init__", return_value=None) as init:
      NVAllocator(dev)

    init.assert_called_once_with(dev, batch_cnt=3)

  def test_host_staging_buffer_gets_bar_view_over_uncached_contiguous_vram(self):
    iface = object.__new__(USBIface)
    iface.pci_dev, iface.vram_bar = MagicMock(), 1
    ret = MagicMock()
    ret.meta.mapping.paddrs, ret.meta.mapping.size, ret.meta.has_cpu_mapping = [(0x123000, 0x200000)], 0x200000, False
    bar_view = iface.pci_dev.map_bar.return_value

    with patch.object(PCIIfaceBase, "alloc", autospec=True, return_value=ret) as alloc:
      self.assertIs(iface.alloc(0x200000, host=True), ret)

    alloc.assert_called_once_with(iface, 0x200000, host=False, uncached=True, cpu_access=False,
                                  contiguous=True, force_devmem=True, cpu_visible=True)
    iface.pci_dev.map_bar.assert_called_once_with(1, off=0x123000, size=0x200000)
    self.assertIs(ret.view, bar_view)
    self.assertFalse(ret.meta.has_cpu_mapping)

  def test_cpu_access_bar_view_is_not_treated_as_process_mapping(self):
    iface = object.__new__(USBIface)
    iface.pci_dev, iface.vram_bar = MagicMock(), 1
    ret = MagicMock()
    ret.meta.has_cpu_mapping = True

    with patch.object(PCIIfaceBase, "alloc", autospec=True, return_value=ret):
      self.assertIs(iface.alloc(0x1000, cpu_access=True), ret)

    self.assertFalse(ret.meta.has_cpu_mapping)

  def test_normal_gpu_buffer_does_not_consume_cpu_visible_vram(self):
    iface = object.__new__(USBIface)
    ret = MagicMock()
    ret.meta.has_cpu_mapping = False

    with patch.object(PCIIfaceBase, "alloc", autospec=True, return_value=ret) as alloc:
      self.assertIs(iface.alloc(0x200000), ret)

    alloc.assert_called_once_with(iface, 0x200000, host=False, uncached=False, cpu_access=False,
                                  contiguous=False, force_devmem=True, cpu_visible=False)


class FakeQueueController:
  def __init__(self):
    self.xdata, self.writes, self.read_arms = {}, [], []
    self.read_data = bytes(ASM24GSPQueueInterface.SRAM_SIZE)

  def read(self, addr, size): return bytes(self.xdata.get(addr+i, 0) for i in range(size))
  def write(self, addr, data):
    self.xdata.update((addr+i, x) for i, x in enumerate(data))
    self.writes.append(("xdata", addr, bytes(data)))
  def scsi_write(self, data, start_slot=0): self.writes.append(("sram", start_slot, bytes(data)))
  def scsi_read_arm(self, size, start_slot=0): self.read_arms.append((size, start_slot))
  def scsi_read(self, size): return memoryview(self.read_data[:size])


class TestASM24GSPQueueInterface(unittest.TestCase):
  def setUp(self):
    self.controller = FakeQueueController()
    self.queue = ASM24GSPQueueInterface(self.controller)

  def test_page_addresses_include_queue_headers_and_contiguous_sram(self):
    paddrs = self.queue.paddrs()
    self.assertEqual(len(paddrs), 129)
    self.assertEqual(paddrs[:6], [0x820000, 0x200000, 0x201000, 0x202000, 0x203000, 0x204000])
    self.assertEqual(paddrs[65:68], [0x240000, 0x241000, 0x242000])
    self.assertEqual(paddrs[-1], 0x27F000)

  def test_writes_route_headers_and_flush_complete_sram_slots(self):
    self.queue[:8] = b"PTE DATA"
    self.queue.view(0x1000)[:4] = b"CMDH"
    self.queue.view(0x2000)[:4] = b"CMDD"
    self.queue.view(0x41000)[:4] = b"STAH"

    self.assertEqual(self.controller.writes[0], ("xdata", 0xA000, b"PTE DATA"))
    self.assertEqual(self.controller.writes[1][0:2], ("sram", 0))
    self.assertEqual(self.controller.writes[1][2][:4], b"CMDH")
    self.assertEqual(self.controller.writes[2][0:2], ("sram", 0))
    self.assertEqual(self.controller.writes[2][2][0x1000:0x1004], b"CMDD")
    self.assertEqual(self.controller.writes[3][0:2], ("sram", 16))
    self.assertEqual(self.controller.writes[3][2][:4], b"STAH")

    self.queue.view(0x80000)[:4] = b"LAST"
    self.assertEqual(self.controller.writes[4][0:2], ("sram", 31))
    self.assertEqual(self.controller.writes[4][2][0x3000:0x3004], b"LAST")

  def test_status_read_uses_armed_bulk_snapshot(self):
    data = bytearray(ASM24GSPQueueInterface.SRAM_SIZE)
    data[0x41000:0x41004] = b"RESP"
    self.controller.read_data = bytes(data)

    self.queue.arm_read()
    self.assertEqual(self.queue.view(0x42000)[:4], b"RESP")
    self.assertEqual(self.controller.read_arms, [(0x80000, 0)])

  def test_nvidia_fixed_page_map_routes_each_queue_region(self):
    queue = ASM24GSPQueueInterface(self.controller, 0xB000, page_paddrs=ASM24GSPQueueInterface.NVIDIA_PAGE_PADDRS)
    self.assertEqual(queue.paddrs(), list(ASM24GSPQueueInterface.NVIDIA_PAGE_PADDRS))

    queue[0:4] = b"PTES"
    queue.view(0x1000)[0:4] = b"CMDH"
    queue.view(0x2000)[0:4] = b"CMD0"
    queue.view(0x6000)[0:4] = b"STAH"
    queue.view(0x7000)[0:4] = b"STA0"
    queue.view(0x8000)[0:4] = b"STA1"

    sram_writes = [write for write in self.controller.writes if write[0] == "sram"]
    self.assertEqual([(write[1], write[2][0x3000:0x3004]) for write in sram_writes],
                     [(4, b"PTES"), (20, b"CMDH"), (19, b"CMD0")])
    xdata_writes = [write for write in self.controller.writes if write[0] == "xdata"]
    self.assertEqual(xdata_writes, [("xdata", 0xB800, b"STAH"), ("xdata", 0xA000, b"STA0"), ("xdata", 0xF000, b"STA1")])

    queue.arm_read()
    queue.sync()
    self.assertEqual(self.controller.read_arms, [])


if __name__ == "__main__": unittest.main()
