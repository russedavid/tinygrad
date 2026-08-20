import ctypes, types, unittest
from unittest.mock import MagicMock

from tinygrad.runtime.autogen import nv
from tinygrad.runtime.support.nv.ip import NVRpcQueue


class TestNVRpcQueueRecords(unittest.TestCase):
  def setUp(self):
    self.queue = object.__new__(NVRpcQueue)
    self.queue.tx = types.SimpleNamespace(msgSize=0x1000, msgCount=4)
    self.storage = bytearray(self.queue.tx.msgSize * self.queue.tx.msgCount)
    self.queue.queue_mv = memoryview(self.storage)
    self.queue._direct_transport, self.queue._direct_status, self.queue._direct_cursor = False, False, None
    self.queue._reserve_direct_credit = False
    self.queue._direct_status_capacity = 2

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

  def test_direct_reader_consumes_initial_two_element_record_with_reserved_credit(self):
    payload = bytes(0x1000)
    raw = self.make_record(0x346, nv.NV_VGPU_MSG_EVENT_GSP_RUN_CPU_SEQUENCER, payload)
    self.write_record(0, raw)
    self.queue.rx_view = [3]
    write_index = getattr(nv.msgqTxHeader, 'writePtr').offset // 4
    self.queue.tx_view = [0] * (write_index + 1)
    self.queue.tx_view[write_index] = 2
    self.queue._direct_status, self.queue._direct_cursor, self.queue._reserve_direct_credit = True, 0, True
    run_cpu_seq = MagicMock()
    self.queue.gsp = types.SimpleNamespace(run_cpu_seq=run_cpu_seq,
      nvdev=types.SimpleNamespace(pci_dev=types.SimpleNamespace(), is_err_state=False, devfmt="test"))

    self.assertEqual(list(self.queue.read_resp()), [(nv.NV_VGPU_MSG_EVENT_GSP_RUN_CPU_SEQUENCER, payload)])
    run_cpu_seq.assert_called_once_with(payload)
    self.assertEqual(self.queue.rx_view[0], 1)
    self.assertEqual(list(self.queue.read_resp()), [])

  def test_direct_reader_rejects_torn_record(self):
    raw = bytearray(self.make_record(0x120, nv.NV_VGPU_MSG_EVENT_GSP_POST_NOCAT_RECORD, bytes(0x100)))
    raw[0x80] ^= 1
    self.write_record(0, raw)
    self.queue.rx_view = [3]
    write_index = getattr(nv.msgqTxHeader, 'writePtr').offset // 4
    self.queue.tx_view = [0] * (write_index + 1)
    self.queue.tx_view[write_index] = 1
    self.queue._direct_status, self.queue._direct_cursor, self.queue._reserve_direct_credit = True, 0, True
    self.queue.gsp = types.SimpleNamespace(
      nvdev=types.SimpleNamespace(pci_dev=types.SimpleNamespace(), is_err_state=False, devfmt="test"))

    with self.assertRaisesRegex(RuntimeError, "checksum"): list(self.queue.read_resp())
    self.assertEqual(self.queue.rx_view[0], 3)

  def test_direct_reader_rejects_record_larger_than_physical_ring(self):
    self.write_record(0, self.make_record(0x120, 0x1234, bytes(0x2000)))
    self.queue._direct_status = True

    with self.assertRaisesRegex(RuntimeError, "needs 3 elements.*2 physical slots"): self.queue._read_record(0)

  def test_direct_reader_consumes_two_element_record_without_pointer_alias(self):
    payload = bytes((x & 0xFF) for x in range(0x1000))
    self.write_record(3, self.make_record(7, 0x1234, payload))
    self.queue.rx_view = [2]
    write_index = getattr(nv.msgqTxHeader, 'writePtr').offset // 4
    self.queue.tx_view = [0] * (write_index + 1)
    self.queue.tx_view[write_index] = 1
    self.queue._direct_status, self.queue._direct_cursor, self.queue._reserve_direct_credit = True, 3, True
    self.queue.gsp = types.SimpleNamespace(nvdev=types.SimpleNamespace(
      pci_dev=types.SimpleNamespace(), is_err_state=False, devfmt="test"))

    self.assertEqual(list(self.queue.read_resp()), [(0x1234, payload)])
    self.assertEqual(self.queue._direct_cursor, 1)
    self.assertEqual(self.queue.rx_view[0], 0)

  def test_direct_reader_consumes_two_records_before_acknowledging(self):
    self.write_record(0, self.make_record(7, 0x1234, b"first"))
    self.write_record(1, self.make_record(8, 0x1234, b"second"))
    writes = []
    class Pointer(list):
      def __setitem__(self, index, value):
        writes.append(value)
        super().__setitem__(index, value)
    self.queue.rx_view = Pointer([3])
    write_index = getattr(nv.msgqTxHeader, 'writePtr').offset // 4
    self.queue.tx_view = [0] * (write_index + 1)
    self.queue.tx_view[write_index] = 2
    self.queue._direct_status, self.queue._direct_cursor, self.queue._reserve_direct_credit = True, 0, True
    self.queue.gsp = types.SimpleNamespace(nvdev=types.SimpleNamespace(
      pci_dev=types.SimpleNamespace(), is_err_state=False, devfmt="test"))

    self.assertEqual(list(self.queue.read_resp()), [(0x1234, b"first"), (0x1234, b"second")])
    self.assertEqual(writes, [1])

  def test_direct_reader_releases_reserved_credit_for_teardown(self):
    self.queue._direct_status, self.queue._direct_cursor, self.queue._reserve_direct_credit, self.queue.rx_view = True, 2, True, [1]

    self.queue.release_direct_credit()

    self.assertEqual(self.queue.rx_view[0], 2)

  def test_direct_reader_returns_all_credit_for_distinct_ring(self):
    self.write_record(0, self.make_record(7, 0x1234, b"response"))
    self.queue.rx_view = [0]
    write_index = getattr(nv.msgqTxHeader, 'writePtr').offset // 4
    self.queue.tx_view = [0] * (write_index + 1)
    self.queue.tx_view[write_index] = 1
    self.queue._direct_status, self.queue._direct_cursor, self.queue._reserve_direct_credit = True, 0, False
    self.queue.gsp = types.SimpleNamespace(nvdev=types.SimpleNamespace(
      pci_dev=types.SimpleNamespace(), is_err_state=False, devfmt="test"))

    self.assertEqual(list(self.queue.read_resp()), [(0x1234, b"response")])
    self.assertEqual(self.queue.rx_view[0], 1)

  def test_reserved_credit_keeps_aliased_ring_collision_free(self):
    count, aliases = 4, (0, 1, 0, 1)
    pending = [(0, 0, count - 1, (), cached) for cached in range(3)]
    seen = set(pending)

    while pending:
      cursor, write, visible_read, records, cached = pending.pop()
      live_slots = [(cursor + offset) % count for offset in range(sum(records))]
      self.assertEqual(len({aliases[slot] for slot in live_slots}), len(live_slots))

      free = (visible_read + count - write - 1) % count
      for size in (1, 2):
        available = cached if size <= cached else free
        if size <= available:
          state = (cursor, (write + size) % count, visible_read, records + (size,), available - size)
          if state not in seen:
            seen.add(state)
            pending.append(state)
      if records:
        next_cursor = (cursor + records[0]) % count
        state = (next_cursor, write, (next_cursor - 1) % count, records[1:], cached)
        if state not in seen:
          seen.add(state)
          pending.append(state)

    self.assertGreater(len(seen), count)

if __name__ == "__main__": unittest.main()
