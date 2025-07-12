import unittest
import torch
from sgmt.model import SGMT

class TestSGMT(unittest.TestCase):
    def test_forward_shape(self):
        src_vocab = ['a','b']
        tgt_vocab = ['A','B']
        model = SGMT(src_vocab, tgt_vocab)
        # Dummy input of length 4 (batch size 1)
        src = torch.randint(0, len(src_vocab), (1,4))
        tgt_in = torch.randint(0, len(tgt_vocab), (1,4))
        logits, routes = model(src, tgt_in)
        # logits should have shape (1, 4, target_vocab_size)
        self.assertEqual(logits.shape, (1, 4, len(tgt_vocab)))
        # routes is the hard selection tensor (batch, slots, modules)
        self.assertIsInstance(routes, torch.Tensor)

if __name__ == "__main__":
    unittest.main()
