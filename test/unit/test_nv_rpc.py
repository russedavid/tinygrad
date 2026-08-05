import ctypes, types, unittest
from unittest.mock import MagicMock

from tinygrad.runtime.autogen import nv
from tinygrad.runtime.support.hcq import MMIOInterface
from tinygrad.runtime.support.nv.ip import NVRpcQueue


class TestNVRpcQueueRecords(unittest.TestCase):
  def setUp(self):
    self.queue = object.__new__(NVRpcQueue)
    self.queue.tx = types.SimpleNamespace(msgSize=0x1000, msgCount=4)
    self.storage = bytearray(self.queue.tx.msgSize * self.queue.tx.msgCount)
    self.queue.queue_mv = memoryview(self.storage)
    self.queue._direct_transport, self.queue._direct_status, self.queue._direct_seen_sequences = False, False, set()

  def make_record(self, sequence:int, function:int, payload:bytes) -> bytes:
    hdr = nv.rpc_message_header_v(signature=nv.NV_VGPU_MSG_SIGNATURE_VALID, function=function,
                                  length=ctypes.sizeof(nv.rpc_message_header_v) + len(payload))
    elem_count = (ctypes.sizeof(nv.GSP_MSG_QUEUE_ELEMENT) + hdr.length + 0xFFF) // 0x1000
    elem = nv.GSP_MSG_QUEUE_ELEMENT(elemCount=elem_count, seqNum=sequence)
    elem.checkSum = self.queue._checksum(bytes(elem) + bytes(hdr) + payload)
    return bytes(elem) + bytes(hdr) + payload

  def write_record(self, slot:int, raw:bytes):
    off = slot * self.queue.tx.msgSize
    first = min(len(raw), len(self.storage) - off)
    self.storage[off:off+first] = raw[:first]
    self.storage[:len(raw)-first] = raw[first:]

  def test_reads_checksum_valid_record_across_ring_wrap(self):
    payload = bytes((x & 0xFF) for x in range(0x1000))
    self.write_record(3, self.make_record(7, 0x1234, payload))

    elem, hdr, raw = self.queue._read_record(3)

    self.assertEqual(elem.elemCount, 2)
    self.assertEqual(hdr.function, 0x1234)
    self.assertEqual(raw[ctypes.sizeof(nv.GSP_MSG_QUEUE_ELEMENT) + ctypes.sizeof(nv.rpc_message_header_v):], payload)

  def test_recovers_existing_sequence_numbers(self):
    self.write_record(0, self.make_record(4, 0x100, b"old"))
    self.write_record(2, self.make_record(9, 0x101, b"new"))

    self.assertEqual(self.queue._valid_sequences(), {4, 9})

  def test_resume_starts_command_sequence_after_existing_records(self):
    storage = (ctypes.c_ubyte * 0x5000)()
    view = MMIOInterface(ctypes.addressof(storage), len(storage))
    view[:ctypes.sizeof(nv.msgqTxHeader)] = bytes(nv.msgqTxHeader(
      size=0x5000, entryOff=0x1000, msgSize=0x1000, msgCount=4, writePtr=3, flags=1,
      rxHdrOff=ctypes.sizeof(nv.msgqTxHeader)))
    for slot, sequence in ((0, 4), (2, 9)):
      raw = self.make_record(sequence, 0x100 + slot, b"record")
      view[0x1000 + slot*0x1000:0x1000 + slot*0x1000 + len(raw)] = raw

    queue = NVRpcQueue(types.SimpleNamespace(), view, resume=True)

    self.assertEqual(queue.seq, 10)

  def test_read_resp_returns_exact_payload_and_advances_by_element_count(self):
    payload = bytes((x & 0xFF) for x in range(0x1000))
    self.write_record(3, self.make_record(9, 0x1234, payload))
    self.queue.rx_view = [3]
    write_index = getattr(nv.msgqTxHeader, 'writePtr').offset // 4
    self.queue.tx_view = [0] * (write_index + 1)
    self.queue.tx_view[write_index] = 1
    self.queue.gsp = types.SimpleNamespace(nvdev=types.SimpleNamespace(pci_dev=types.SimpleNamespace(), is_err_state=False, devfmt="test"))

    self.assertEqual(list(self.queue.read_resp()), [(0x1234, payload)])
    self.assertEqual(self.queue.rx_view[0], 1)

  def test_direct_reader_recovers_record_when_wrapped_pointer_looks_empty(self):
    payload = bytes(0x1000)
    raw = self.make_record(0x346, nv.NV_VGPU_MSG_EVENT_GSP_RUN_CPU_SEQUENCER, payload)
    self.write_record(0, raw)
    self.write_record(2, raw)
    self.queue.rx_view = [0]
    write_index = getattr(nv.msgqTxHeader, 'writePtr').offset // 4
    self.queue.tx_view = [0] * (write_index + 1)
    self.queue._direct_status, self.queue._direct_seen_sequences = True, set()
    run_cpu_seq = MagicMock()
    self.queue.gsp = types.SimpleNamespace(run_cpu_seq=run_cpu_seq,
      nvdev=types.SimpleNamespace(pci_dev=types.SimpleNamespace(), is_err_state=False, devfmt="test"))

    self.assertEqual(list(self.queue.read_resp()), [(nv.NV_VGPU_MSG_EVENT_GSP_RUN_CPU_SEQUENCER, payload)])
    run_cpu_seq.assert_called_once_with(payload)
    self.assertEqual(list(self.queue.read_resp()), [])

  def test_direct_reader_acknowledges_torn_nocat_record(self):
    raw = bytearray(self.make_record(0x120, nv.NV_VGPU_MSG_EVENT_GSP_POST_NOCAT_RECORD, bytes(0x100)))
    raw[0x80] ^= 1
    self.write_record(1, raw)
    self.queue.rx_view = [1]
    write_index = getattr(nv.msgqTxHeader, 'writePtr').offset // 4
    self.queue.tx_view = [0] * (write_index + 1)
    self.queue.tx_view[write_index] = 2
    self.queue._direct_status, self.queue._direct_seen_sequences = True, set()
    self.queue.gsp = types.SimpleNamespace(
      nvdev=types.SimpleNamespace(pci_dev=types.SimpleNamespace(), is_err_state=False, devfmt="test"))

    self.assertEqual(list(self.queue.read_resp()), [])
    self.assertEqual(self.queue.rx_view[0], 2)


if __name__ == "__main__": unittest.main()
