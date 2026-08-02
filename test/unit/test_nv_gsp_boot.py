import struct, types, unittest
from unittest.mock import MagicMock, patch

from tinygrad.runtime.autogen import nv
from tinygrad.runtime.support.nv.ip import NV_FLCN, NV_GSP
from tinygrad.runtime.support.system import USBPCIDevice
from tinygrad.runtime.support.usb import CustomASM24Controller


class TestNVGSPSRAMBoot(unittest.TestCase):
  def test_sram_boot_does_not_stage_duplicate_vram_sources(self):
    gsp = object.__new__(NV_GSP)
    vram_size = 24 << 30
    gsp.nvdev = types.SimpleNamespace(
      chip_name="GA102", fw_name="ga102", vram_size=vram_size, fmc_boot=False,
      flcn=types.SimpleNamespace(frts_offset=vram_size - (2 << 20)),
      pci_dev=types.SimpleNamespace(gsp_sram_boot=True), _alloc_boot_mem=MagicMock())
    sections = [types.SimpleNamespace(name=".fwimage", content=b"I" * 0x2000),
                types.SimpleNamespace(name=".fwsignature_ga10x", content=b"S" * 0x1000)]
    fw_header = types.SimpleNamespace(data_offset=4, data_size=4, header_offset=0)
    booter_desc = types.SimpleNamespace(monitorCodeOffset=1, monitorDataOffset=2, manifestOffset=3)

    with patch("tinygrad.runtime.support.nv.ip.fetch_fw", side_effect=[b"elf", bytes(4) + b"BOOT"]), \
         patch("tinygrad.runtime.support.nv.ip.elf_loader", return_value=(None, sections, None)), \
         patch("tinygrad.runtime.support.nv.ip.nv.struct_nvfw_bin_hdr") as header_t, \
         patch("tinygrad.runtime.support.nv.ip.nv.RM_RISCV_UCODE_DESC") as desc_t:
      header_t.from_buffer_copy.return_value, desc_t.from_buffer_copy.return_value = fw_header, booter_desc
      gsp.init_wpr_meta()

    gsp.nvdev._alloc_boot_mem.assert_not_called()
    self.assertEqual(gsp.wpr_meta_sysmem, 0x200000)
    meta = nv.GspFwWprMeta.from_buffer_copy(gsp._boot_sram)
    self.assertEqual((meta.sysmemAddrOfSignature, meta.sysmemAddrOfBootloader, meta.sysmemAddrOfRadix3Elf),
                     (0x201000, 0x202000, 0x208000))

  def test_sram_wpr_uses_cyclic_84_page_image_ring(self):
    gsp = object.__new__(NV_GSP)
    gsp.gsp_image = b''.join(bytes((page,)) * 0x1000 for page in range(86))
    gsp.gsp_signature, gsp.booter_image = b'S' * 0x1000, b'B' * 0x5800

    image = gsp._build_sram_wpr(nv.GspFwWprMeta())

    self.assertEqual(len(image), 0x80000)
    meta = nv.GspFwWprMeta.from_buffer_copy(image)
    self.assertEqual((meta.sysmemAddrOfSignature, meta.sysmemAddrOfBootloader, meta.sysmemAddrOfRadix3Elf),
                     (0x201000, 0x202000, 0x208000))
    self.assertEqual(image[0x1000:0x2000], gsp.gsp_signature)
    self.assertEqual(image[0x2000:0x7800], gsp.booter_image)
    image_ptes = struct.unpack_from('<86Q', image, 0xA000)
    self.assertEqual(image_ptes[:2], (0x22C000, 0x22D000))
    self.assertEqual(image_ptes[83:86], (0x27F000, 0x22C000, 0x22D000))
    self.assertEqual(image[0x2C000:0x2D000], bytes(0x1000))
    self.assertEqual(image[0x2D000:0x2E000], bytes((1,)) * 0x1000)

  def test_stream_schedule_rotates_across_all_ring_slots(self):
    ring_size = CustomASM24Controller.GSP_RING_PAGES * 0x1000
    batch_size = CustomASM24Controller.GSP_STREAM_BATCH_PAGES * 0x1000
    chunks = list(CustomASM24Controller.gsp_stream_chunks(bytes(ring_size + 4 * batch_size - 17)))

    self.assertEqual([slot for _, slot, _ in chunks], [11, 18, 25, 11])
    self.assertEqual([len(payload) for _, _, payload in chunks], [batch_size] * 4)
    self.assertAlmostEqual(chunks[0][0], 0.003)
    self.assertAlmostEqual(chunks[3][0], 0.0044)
    self.assertEqual(chunks[-1][2][-17:], bytes(17))

  def test_transport_restores_queues_and_argument_windows(self):
    dev = object.__new__(USBPCIDevice)
    dev.usb, dev._wait_until = MagicMock(), MagicMock()
    dev.gsp_queues = types.SimpleNamespace(_root=types.SimpleNamespace(_mirror=bytearray(b'queue')))
    dev._gsp_args = {0x100:b'rm', 0x200:b'libos'}

    dev.stream_gsp_boot(b'image', 4.0)

    dev.usb.stream_gsp_image.assert_called_once_with(b'image', 4.0)
    self.assertEqual(dev._wait_until.call_args_list[0].args, (4.270,))
    self.assertEqual(dev._wait_until.call_args_list[1].args, (4.380,))
    self.assertEqual(dev.usb.scsi_write.call_args_list[0].args, (b'queue',))
    self.assertEqual(dev.usb.scsi_write.call_args_list[1].args, (b'queue',))
    self.assertEqual(dev.usb.write.call_args_list[0].args, (0xB900, b'rm'))
    self.assertEqual(dev.usb.write.call_args_list[1].args, (0xBA00, b'libos'))

  @patch('tinygrad.runtime.support.nv.ip.time.perf_counter', return_value=12.5)
  def test_sec2_launch_sets_local_contexts_then_streams(self, _):
    flcn = object.__new__(NV_FLCN)
    contexts = [MagicMock() for _ in range(8)]
    context_regs = MagicMock()
    context_regs.__getitem__.side_effect = contexts.__getitem__
    context_reg = MagicMock()
    context_reg.with_base.return_value = context_regs
    cpuctl = MagicMock()
    cpuctl.with_base.return_value.read_bitfields.return_value = {'alias_en': 0}
    stream, invalidate = MagicMock(), MagicMock()
    flcn.nvdev = types.SimpleNamespace(
      pci_dev=types.SimpleNamespace(gsp_sram_boot=True, stream_gsp_boot=stream),
      gsp=types.SimpleNamespace(gsp_image=b'image', invalidate_rpc_memory=invalidate),
      NV_PFALCON_FBIF_TRANSCFG=context_reg, NV_PFALCON_FBIF_TRANSCFG_MEM_TYPE_PHYSICAL=7,
      NV_PFALCON_FALCON_CPUCTL=cpuctl, NV_PFALCON_FALCON_CPUCTL_ALIAS=0x130)
    flcn.sec2 = 0x840000

    flcn.start_cpu(flcn.sec2)

    for context in contexts: context.update.assert_called_once_with(target=0, mem_type=7)
    cpuctl.with_base.return_value.write.assert_called_once_with(startcpu=1)
    stream.assert_called_once_with(b'image', 12.5)
    invalidate.assert_called_once_with()

  def test_rearms_sec2_for_cpu_sequencer_reuse(self):
    flcn = object.__new__(NV_FLCN)
    context = MagicMock()
    contexts = MagicMock()
    contexts.__getitem__.return_value = context
    context_reg = MagicMock()
    context_reg.with_base.return_value = contexts
    flcn.reset, flcn.disable_ctx_req = MagicMock(), MagicMock()
    flcn.sec2 = 0x840000
    flcn.nvdev = types.SimpleNamespace(
      pci_dev=types.SimpleNamespace(gsp_sram_boot=True), NV_PFALCON_FBIF_TRANSCFG=context_reg,
      NV_PFALCON_FBIF_TRANSCFG_TARGET_COHERENT_SYSMEM=5, NV_PFALCON_FBIF_TRANSCFG_MEM_TYPE_PHYSICAL=7)

    flcn.rearm_sec2_queue()

    flcn.reset.assert_called_once_with(0x840000)
    flcn.disable_ctx_req.assert_called_once_with(0x840000)
    context.update.assert_called_once_with(target=5, mem_type=7)


if __name__ == '__main__': unittest.main()
