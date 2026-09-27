"""Full-trainer test of weighted NNZ forward with the validated dense backward."""
import os,runpy,sys
sys.path.insert(0,'/workspace/nnue');sys.path.insert(0,'/workspace')
import torch
from model.modules import grouped_linear as grouped
import sparse_weighted as sparse

if os.environ.get('NNUE_SPARSE','1') == '1':
    def forward(ctx,x,weight,bias,indices):
        x=x.contiguous();weight=weight.contiguous();indices=indices.flatten().to(torch.int64).contiguous()
        rows,counts=grouped._route_rows(indices)
        ctx.save_for_backward(x,weight,rows,counts)
        return sparse.forward(x,sparse.transpose(weight),bias.contiguous(),indices,128,'compact')
    grouped._GroupedLinear.forward=staticmethod(forward)

if os.environ.get('NNUE_SPARSE_TEST') == '1':
    import pytest
    sys.exit(pytest.main(['-q','tests/test_grouped_linear.py']))
runpy.run_path('/workspace/nnue/tests/bench_training.py',run_name='__main__')
